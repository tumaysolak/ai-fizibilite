#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GPU / AI FİZİBİLİTE HESAP MOTORU (engine.py)  —  v4 (düzeltilmiş)
================================================================================
Bu dosya, orijinal "Opt.py" (v3) hesap modelinin İNCELEME RAPORUNDA tespit
edilen tüm hataları düzelten temiz sürümüdür. Web fizibilite platformunun
(app.py) çekirdeğini oluşturur.

Uygulanan düzeltmeler (rapordaki numaralarla):
  A1  Decode hızına KV-cache okuması eklendi (uzun bağlamda gerçekçi TPS).
  A2  Diffusion modelleri "token/sn" yerine "görsel/sn" birimiyle ele alınır;
      token maliyeti üretilmez.
  A3  Aktivasyon belleği artık batch × seq × hidden × katman ile ölçekleniyor
      (ağırlık boyutuna bağlı sahte orandan vazgeçildi).
  A4  Throughput (data-parallel toplam) ile tek-istek gecikmesi AÇIKÇA ayrıldı.
  A5  KV-cache tüm agresif kuantlarda (INT4/INT8/GGUF/IQ) FP16'ya sabitlenir.
  A6  Sihirli sabitler parametreleştirildi (güç referansı vb.).
  B1  Kısıt (min-tps/max-power) sağlanamazsa SESSİZCE düşülmez; sonuç
      "constraints_met=False" ve açık uyarı ile işaretlenir.
  B2  MILP ve heuristic aynı VRAM muhasebesini kullanır (ham VRAM).
  B3  NVLink/çoklu-GPU cezası VRAM'e değil THROUGHPUT'a uygulanır.
  B4  Çok-düğüm (multi-node) verimlilik kaybı MILP min-tps kontrolüne dahildir.
  B5  Offload aktivasyon transferi gerçek batch'i kullanır.
  C1  TPS=0 durumunda NaN yerine None döner.
  C2  Model/GPU aramada tam-ad (exact) önceliği + substring yedeği.
  C4  "estimated_tps or ..." tuzağı yerine "is None" kontrolü.
================================================================================
"""
from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# SABİTLER
# ==============================================================================
BYTES_PER_GB: float = 1024 ** 3
SECONDS_PER_HOUR: int = 3600
HOURS_PER_MONTH: float = 730.0

DEFAULT_HARDWARE_LIFETIME_HOURS: float = 3 * 365 * 24 * 0.85  # 3 yıl, %85 uptime

# VRAM overhead katsayıları
FRAMEWORK_OVERHEAD_ALPHA: float = 0.06   # framework/CUDA context payı
SAFETY_MARGIN_M: float = 0.18            # güvenlik marjı
ACTIVATION_BYTES_PER_ELEM: float = 2.0   # aktivasyon fp16 varsayımı
ACTIVATION_LAYER_FACTOR: float = 2.0     # katman içi ara tensör çarpanı (kaba)

# Throughput verimliliği
DEFAULT_ETA_MEM: float = 0.55            # memory-bound gerçek verim
DEFAULT_ETA_COMPUTE: float = 0.35        # compute-bound gerçek verim (düşük MFU)
FLOPS_PER_PARAM_PER_TOKEN: float = 2.0

# Çoklu-GPU / düğüm (THROUGHPUT tarafında, VRAM'de DEĞİL — düzeltme B3)
GAMMA_WITH_NVLINK: float = 0.90
GAMMA_NO_NVLINK: float = 0.75
MAX_GPUS_PER_NODE: int = 8
INTERNODE_LOSS_PER_EXTRA_NODE: float = 0.15
MIN_INTERNODE_FLOOR: float = 0.25
MAX_GPUS_WITHOUT_NVLINK: int = 4          # fiziksel gerçekçilik kısıtı

# Offload (CPU-GPU hibrit)
SYSTEM_RAM_BANDWIDTH_GBPS: float = 50.0
CPU_COMPUTE_PENALTY: float = 8.0

# Darboğaz referansları (A6 — parametreleştirildi)
POWER_REFERENCE_WATTS: float = 400.0
DEFAULT_NVME_READ_SPEED_GBPS: float = 3.5


# ==============================================================================
# ENUM'LAR
# ==============================================================================
class DeploymentType(enum.Enum):
    LOCAL = "Yerel / On-Premise"
    CLOUD = "Bulut (API)"


class ArchitectureType(enum.Enum):
    DENSE_TRANSFORMER = "Yoğun (Dense) Transformer"
    MOE_TRANSFORMER = "Mixture-of-Experts (MoE)"
    ENCODER_DECODER = "Encoder-Decoder (örn. Whisper)"
    DIFFUSION = "Diffusion (Görsel Üretim)"
    CLOUD_OPAQUE = "Bulut / Mimari Kapalı"


class GPUTier(enum.Enum):
    CONSUMER = "Tüketici"
    WORKSTATION = "İş İstasyonu"
    DATACENTER = "Veri Merkezi"


class BottleneckType(enum.Enum):
    VRAM = "VRAM Sınırı"
    MEMORY_BANDWIDTH = "Bellek Bant Genişliği (Memory-Bound)"
    COMPUTE_BOUND = "İşlem Gücü Sınırı (Compute-Bound)"
    KV_CACHE_BOUND = "KV-Cache Bant Genişliği (Uzun Bağlam)"
    NVLINK_REQUIREMENT = "NVLink / Çoklu-GPU Gecikmesi"
    POWER_DRAW = "Yüksek Güç Tüketimi"
    CPU_OFFLOAD_PENALTY = "CPU Offload Performans Kaybı"
    NETWORK_API = "İnternet / API Rate-Limit"
    NONE = "Belirgin Darboğaz Yok"


# ==============================================================================
# KUANTİZASYON FORMATLARI
# ==============================================================================
@dataclass(frozen=True)
class QuantFormat:
    key: str
    label: str
    bytes_per_param: float
    description: str = ""


QUANT_FORMATS: Dict[str, QuantFormat] = {
    "FP32": QuantFormat("FP32", "FP32", 4.0, "Tam hassasiyet"),
    "FP16": QuantFormat("FP16", "FP16", 2.0, "Yarı hassasiyet"),
    "BF16": QuantFormat("BF16", "BF16", 2.0, "Brain Float16"),
    "FP8": QuantFormat("FP8", "FP8", 1.0, "H100 tensor-core FP8"),
    "INT8": QuantFormat("INT8", "INT8", 1.0, "8-bit tamsayı"),
    "INT4": QuantFormat("INT4", "INT4", 0.5, "4-bit tamsayı (GPTQ/AWQ)"),
    "Q8_0": QuantFormat("Q8_0", "GGUF Q8_0", 1.0625, "Neredeyse kayıpsız 8-bit"),
    "Q6_K": QuantFormat("Q6_K", "GGUF Q6_K", 0.820, "~6.56 bit/param"),
    "Q5_K_M": QuantFormat("Q5_K_M", "GGUF Q5_K_M", 0.694, "~5.55 bit/param"),
    "Q4_K_M": QuantFormat("Q4_K_M", "GGUF Q4_K_M", 0.604, "~4.83 bit/param — popüler denge"),
    "Q3_K_M": QuantFormat("Q3_K_M", "GGUF Q3_K_M", 0.494, "~3.95 bit/param"),
    "IQ4_XS": QuantFormat("IQ4_XS", "GGUF IQ4_XS", 0.535, "~4.28 bit/param"),
    "CLOUD": QuantFormat("CLOUD", "N/A (Bulut)", 0.0, "Bulut API"),
}

# A5: KV-cache bu formatlarda FP16'ya sabitlenir (kalite için pratik norm)
_AGGRESSIVE_QUANT_KEYS = {"INT8", "INT4", "FP8", "Q8_0", "Q6_K", "Q5_K_M",
                         "Q4_K_M", "Q3_K_M", "IQ4_XS"}


def get_quant_format(key: str) -> QuantFormat:
    if key not in QUANT_FORMATS:
        raise KeyError(f"Bilinmeyen kuantizasyon: '{key}'")
    return QUANT_FORMATS[key]


# ==============================================================================
# VERİ MODELLERİ
# ==============================================================================
@dataclass(frozen=True)
class GPUSpec:
    name: str
    tier: GPUTier
    vram_gb: float
    memory_bandwidth_gbps: float
    fp16_tflops: float
    price_usd: float
    power_watts: float
    pcie_bandwidth_gbps: float
    supports_nvlink: bool
    nvlink_bandwidth_gbps: float = 0.0
    source_note: str = ""

    def __str__(self) -> str:
        return f"{self.name} ({self.vram_gb:.0f}GB)"


@dataclass(frozen=True)
class AIModelSpec:
    name: str
    use_case: str
    deployment: DeploymentType
    architecture: ArchitectureType
    precision: str
    params_billions: float
    active_params_billions: Optional[float] = None
    num_layers: Optional[int] = None
    hidden_dim: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    default_context_length: int = 4096
    disk_size_gb: Optional[float] = None
    known_bottleneck_hint: Optional[BottleneckType] = None
    cloud_token_cost_usd_per_1k: Optional[float] = None
    notes: str = ""

    def quant(self) -> QuantFormat:
        return get_quant_format(self.precision)

    def effective_active_params_billions(self) -> float:
        return self.active_params_billions if self.active_params_billions is not None else self.params_billions

    def effective_kv_dim(self) -> Optional[int]:
        """GQA desteği: KV başlıkları query başlıklarından az olabilir."""
        if self.hidden_dim is None:
            return None
        if self.num_kv_heads is not None and self.num_attention_heads:
            head_dim = self.hidden_dim / self.num_attention_heads
            return int(self.num_kv_heads * head_dim)
        return self.hidden_dim

    def is_local(self) -> bool:
        return self.deployment == DeploymentType.LOCAL


# ==============================================================================
# KATALOG YÜKLEME
# ==============================================================================
def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gpu_catalog(path: Optional[str] = None) -> List[GPUSpec]:
    path = path or os.path.join(SCRIPT_DIR, "data", "gpu_catalog.json")
    raw = _load_json(path)
    out = []
    for it in raw["gpus"]:
        out.append(GPUSpec(
            name=it["name"], tier=GPUTier[it["tier"]],
            vram_gb=float(it["vram_gb"]), memory_bandwidth_gbps=float(it["memory_bandwidth_gbps"]),
            fp16_tflops=float(it["fp16_tflops"]), price_usd=float(it["price_usd"]),
            power_watts=float(it["power_watts"]), pcie_bandwidth_gbps=float(it["pcie_bandwidth_gbps"]),
            supports_nvlink=bool(it["supports_nvlink"]),
            nvlink_bandwidth_gbps=float(it.get("nvlink_bandwidth_gbps", 0.0)),
            source_note=it.get("source_note", ""),
        ))
    return out


def load_model_catalog(path: Optional[str] = None) -> List[AIModelSpec]:
    path = path or os.path.join(SCRIPT_DIR, "data", "model_catalog.json")
    raw = _load_json(path)
    out = []
    for it in raw["models"]:
        hint = it.get("known_bottleneck_hint")
        out.append(AIModelSpec(
            name=it["name"], use_case=it["use_case"],
            deployment=DeploymentType[it["deployment"]],
            architecture=ArchitectureType[it["architecture"]],
            precision=it["precision"], params_billions=float(it["params_billions"]),
            active_params_billions=(float(it["active_params_billions"])
                                    if it.get("active_params_billions") is not None else None),
            num_layers=it.get("num_layers"), hidden_dim=it.get("hidden_dim"),
            num_attention_heads=it.get("num_attention_heads"), num_kv_heads=it.get("num_kv_heads"),
            default_context_length=int(it.get("default_context_length", 4096)),
            disk_size_gb=(float(it["disk_size_gb"]) if it.get("disk_size_gb") is not None else None),
            known_bottleneck_hint=(BottleneckType[hint] if hint else None),
            cloud_token_cost_usd_per_1k=it.get("cloud_token_cost_usd_per_1k"),
            notes=it.get("notes", ""),
        ))
    return out


def _find_by_name(name: str, items, get_name) -> Optional[Any]:
    """C2: önce tam (case-insensitive) eşleşme, yoksa substring."""
    nl = name.strip().lower()
    for it in items:
        if get_name(it).lower() == nl:
            return it
    for it in items:
        if nl in get_name(it).lower():
            return it
    return None


def find_model_by_name(name: str, catalog: List[AIModelSpec]) -> Optional[AIModelSpec]:
    return _find_by_name(name, catalog, lambda m: m.name)


def find_gpu_by_name(name: str, catalog: List[GPUSpec]) -> Optional[GPUSpec]:
    return _find_by_name(name, catalog, lambda g: g.name)


# ==============================================================================
# VRAM HESAPLAMA (A3 düzeltmesi: aktivasyon batch/seq'e bağlı)
# ==============================================================================
@dataclass
class VRAMBreakdown:
    weight_memory_gb: float
    kv_cache_memory_gb: float
    activation_memory_gb: float
    subtotal_gb: float
    safety_margin_gb: float
    total_required_gb: float
    formula_trace: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class VRAMCalculator:
    def __init__(self, alpha=FRAMEWORK_OVERHEAD_ALPHA, margin=SAFETY_MARGIN_M):
        self.alpha = alpha
        self.margin = margin

    def weight_memory_gb(self, model: AIModelSpec) -> float:
        # MoE dahil TOPLAM parametre (tüm expert'ler bellekte durur)
        raw = model.params_billions * 1e9 * model.quant().bytes_per_param
        return raw * (1 + self.alpha) / BYTES_PER_GB

    def _kv_bytes_per_param(self, model: AIModelSpec, kv_override: Optional[str]) -> float:
        key = kv_override or model.precision
        if key in _AGGRESSIVE_QUANT_KEYS:  # A5: tüm agresif kuantlarda KV = FP16
            key = "FP16"
        return get_quant_format(key).bytes_per_param

    def kv_cache_memory_gb(self, model, context_length=None, batch_size=1, kv_override=None) -> float:
        if model.architecture in (ArchitectureType.DIFFUSION, ArchitectureType.CLOUD_OPAQUE):
            return 0.0
        if model.num_layers is None or model.hidden_dim is None:
            return 0.0
        seq = context_length if context_length is not None else model.default_context_length
        if seq <= 0 or batch_size <= 0:
            return 0.0
        bpp = self._kv_bytes_per_param(model, kv_override)
        arch_corr = 0.5 if model.architecture == ArchitectureType.ENCODER_DECODER else 1.0
        kv_dim = model.effective_kv_dim() or model.hidden_dim
        raw = 2.0 * model.num_layers * kv_dim * seq * batch_size * bpp * arch_corr
        return raw / BYTES_PER_GB

    def activation_memory_gb(self, model, context_length=None, batch_size=1) -> float:
        """A3: aktivasyon batch×seq×hidden ile ölçeklenir. Çıkarımda katmanlar
        SIRALI işlendiği için num_layers ile ÇARPILMAZ (aynı anda tek katmanın
        ara tampon(lar)ı tutulur); ACTIVATION_LAYER_FACTOR bu tamponların kaba
        çarpanıdır. Diffusion gibi hidden_dim'i olmayan modellerde ağırlığın
        küçük bir payı kullanılır."""
        if model.architecture == ArchitectureType.CLOUD_OPAQUE:
            return 0.0
        if model.hidden_dim is None:
            return self.weight_memory_gb(model) * 0.10
        d = model.hidden_dim
        seq = context_length if context_length is not None else model.default_context_length
        seq = max(seq, 1)
        raw = batch_size * seq * d * ACTIVATION_BYTES_PER_ELEM * ACTIVATION_LAYER_FACTOR
        return raw / BYTES_PER_GB

    def total_required(self, model, context_length=None, batch_size=1, kv_override=None) -> VRAMBreakdown:
        trace = []
        w = self.weight_memory_gb(model)
        trace.append(f"Ağırlık = {model.params_billions:.2f}B × {model.quant().bytes_per_param} "
                     f"byte ({model.quant().label}) × (1+{self.alpha}) = {w:.2f} GB")
        kv = self.kv_cache_memory_gb(model, context_length, batch_size, kv_override)
        kv_dim_used = model.effective_kv_dim() or model.hidden_dim or 0
        gqa = (f" [GQA d_kv={kv_dim_used}<d={model.hidden_dim}]"
               if model.num_kv_heads is not None and model.hidden_dim else "")
        trace.append(f"KV-cache = 2×L×d_kv×S×B×byte = {kv:.2f} GB{gqa}")
        act = self.activation_memory_gb(model, context_length, batch_size)
        trace.append(f"Aktivasyon (B={batch_size},S={context_length or model.default_context_length}) = {act:.2f} GB")
        subtotal = w + kv + act
        margin = subtotal * self.margin
        total = subtotal + margin
        trace.append(f"Ara toplam = {subtotal:.2f} GB, güvenlik marjı %{self.margin*100:.0f} = {margin:.2f} GB")
        trace.append(f"TOPLAM gerekli VRAM = {total:.2f} GB")
        return VRAMBreakdown(round(w, 3), round(kv, 3), round(act, 3),
                             round(subtotal, 3), round(margin, 3), round(total, 3), trace)


# ==============================================================================
# THROUGHPUT (A1: KV-cache okuması dahil, A2: diffusion ayrı, A4: latency/throughput ayrımı)
# ==============================================================================
@dataclass
class ThroughputResult:
    unit: str                       # "token/s" veya "image/s"
    single_stream_tps: Optional[float]   # tek istek hızı (latency temelli), None=uygulanamaz
    aggregate_tps: Optional[float]       # çoklu-GPU toplam throughput (data-parallel)
    regime: str                     # memory-bound / compute-bound / kv-bound / n-a
    kv_fraction: float              # decode okumasında KV payı (0-1)
    explanation: str


CALIBRATION_TABLE: Dict[Tuple[str, str], float] = {}


def load_calibration_file(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    for e in entries:
        CALIBRATION_TABLE[(e["gpu"], e["model"])] = float(e["measured_tps"])
    return len(entries)


def _calibrated_tps(model: AIModelSpec, gpu: GPUSpec) -> Optional[float]:
    for (gk, mk), tps in CALIBRATION_TABLE.items():
        if gk.lower() in gpu.name.lower() and mk.lower() in model.name.lower():
            return tps
    return None


class ThroughputEstimator:
    def __init__(self, eta_mem=DEFAULT_ETA_MEM, eta_compute=DEFAULT_ETA_COMPUTE, use_calibration=True):
        self.eta_mem = eta_mem
        self.eta_compute = eta_compute
        self.use_calibration = use_calibration

    def single_stream(self, model, gpu, context_length=None, batch_size=1) -> ThroughputResult:
        """Tek isteğin token üretim hızı. A1: decode adımında KV-cache de okunur."""
        if model.architecture == ArchitectureType.CLOUD_OPAQUE:
            return ThroughputResult("token/s", None, None, "n-a", 0.0,
                                    "Bulut modeli: yerel hız hesaplanmaz (sağlayıcıya bağlı).")
        if model.architecture == ArchitectureType.DIFFUSION:
            # A2: diffusion → görsel/sn (kaba: sabit adım sayısı × FLOP/adım)
            return self._diffusion(model, gpu)

        # Kalibrasyon önceliği
        if self.use_calibration:
            cal = _calibrated_tps(model, gpu)
            if cal is not None:
                return ThroughputResult("token/s", cal, cal, "kalibre", 0.0,
                                        f"Kalibrasyon tablosundan ölçülen TPS: {cal:.1f}")

        n_active = model.effective_active_params_billions() * 1e9
        bpp = model.quant().bytes_per_param
        weight_bytes = n_active * bpp

        # A1: decode adımı başına KV-cache okuması
        seq = context_length if context_length is not None else model.default_context_length
        kv_bytes = 0.0
        if model.num_layers and model.hidden_dim and seq > 0:
            kv_dim = model.effective_kv_dim() or model.hidden_dim
            kv_bpp = get_quant_format("FP16").bytes_per_param  # A5
            kv_bytes = 2.0 * model.num_layers * kv_dim * seq * kv_bpp

        bytes_per_token = weight_bytes + kv_bytes
        bw = gpu.memory_bandwidth_gbps * BYTES_PER_GB
        peak = gpu.fp16_tflops * 1e12

        t_mem = bytes_per_token / (bw * self.eta_mem)
        t_compute = (FLOPS_PER_PARAM_PER_TOKEN * n_active * batch_size) / (peak * self.eta_compute)
        t_step = max(t_mem, t_compute)
        tps = 1.0 / t_step if t_step > 0 else None

        kv_frac = kv_bytes / bytes_per_token if bytes_per_token > 0 else 0.0
        if t_compute > t_mem:
            regime = "compute-bound"
        elif kv_frac > 0.5:
            regime = "kv-bound"
        else:
            regime = "memory-bound"
        expl = (f"Adım/token okuması: ağırlık {weight_bytes/1e9:.2f}GB + KV {kv_bytes/1e9:.2f}GB "
                f"(KV payı %{kv_frac*100:.0f}) → {regime}. Tek istek ~{tps:.1f} token/sn." if tps else
                "Hız hesaplanamadı.")
        return ThroughputResult("token/s", round(tps, 2) if tps else None,
                                round(tps, 2) if tps else None, regime, round(kv_frac, 3), expl)

    def _diffusion(self, model, gpu) -> ThroughputResult:
        # Kaba model: 1 görsel ≈ steps × 2N FLOP (N=param). SDXL ~30 adım varsayımı.
        steps = 30
        n = model.params_billions * 1e9
        flops_per_image = steps * FLOPS_PER_PARAM_PER_TOKEN * n * 2  # UNet iki geçiş (cfg)
        peak = gpu.fp16_tflops * 1e12
        img_per_s = (peak * self.eta_compute) / flops_per_image if flops_per_image > 0 else None
        expl = (f"Diffusion: ~{steps} adım/görsel, tahmini {img_per_s:.2f} görsel/sn "
                f"(compute-bound; token birimi UYGULANMAZ)." if img_per_s else "Hesaplanamadı.")
        return ThroughputResult("image/s", round(img_per_s, 3) if img_per_s else None,
                                round(img_per_s, 3) if img_per_s else None, "compute-bound", 0.0, expl)

    def aggregate(self, model, gpu, num_gpus, single: ThroughputResult, num_nodes=1) -> ThroughputResult:
        """A4/B3/B4: çoklu-GPU TOPLAM throughput (data-parallel). Ölçekleme VRAM'e
        değil throughput'a uygulanır; çok-düğüm cezası da burada uygulanır."""
        base = single.single_stream_tps
        if base is None:
            return single
        if num_gpus <= 1:
            agg = base
        else:
            gamma = GAMMA_WITH_NVLINK if gpu.supports_nvlink else GAMMA_NO_NVLINK
            agg = base * (1 + (num_gpus - 1) * gamma)
        if num_nodes > 1:
            eff = max(MIN_INTERNODE_FLOOR, 1 - INTERNODE_LOSS_PER_EXTRA_NODE * (num_nodes - 1))
            agg *= eff
        return ThroughputResult(single.unit, base, round(agg, 2), single.regime,
                                single.kv_fraction,
                                single.explanation + f" | {num_gpus} GPU toplam ~{agg:.1f} {single.unit}.")

    def affine_params(self, model, gpu, context_length=None) -> Tuple[float, float]:
        """MILP için: aggregate_tps(n) = const*y + slope*n biçiminde katsayılar."""
        s = self.single_stream(model, gpu, context_length)
        base = s.single_stream_tps or 0.0
        gamma = GAMMA_WITH_NVLINK if gpu.supports_nvlink else GAMMA_NO_NVLINK
        return base * (1 - gamma), base * gamma


# ==============================================================================
# GPU SEÇİM — MILP (B1/B2/B3/B4 düzeltmeleri)
# ==============================================================================
@dataclass
class GPUConfiguration:
    gpu: GPUSpec
    num_gpus: int
    num_nodes: int
    total_vram_gb: float
    is_feasible: bool
    constraints_met: bool          # B1: min-tps/max-power sağlandı mı?
    total_cost_usd: float
    estimated_tps: Optional[float]
    tps_unit: str
    solver_used: str
    warnings: List[str] = field(default_factory=list)

    def label(self) -> str:
        prefix = f"{self.num_gpus}× " if self.num_gpus != 1 else "1× "
        node = f" [{self.num_nodes} düğüm]" if self.num_nodes > 1 else ""
        return prefix + self.gpu.name + node


class MILPGPUSelector:
    def __init__(self, gpu_catalog, throughput_est: ThroughputEstimator, max_gpus_per_type=16):
        self.catalog = gpu_catalog
        self.tp = throughput_est
        self.max_per_type = max_gpus_per_type

    def solve(self, model, required_vram_gb, context_length=None,
              min_tps=None, max_power_watts=None) -> Optional[GPUConfiguration]:
        k = len(self.catalog)
        if k == 0:
            return None
        big_m = float(self.max_per_type)
        c = np.concatenate([np.array([g.price_usd for g in self.catalog]), np.zeros(k)])
        cons = []

        # (1) n_g <= M*y_g
        A1 = np.zeros((k, 2 * k))
        for i in range(k):
            A1[i, i] = 1.0
            A1[i, k + i] = -big_m
        cons.append(LinearConstraint(A1, -np.inf, 0.0))
        # (2) sum(y_g)=1 (homojen filo)
        A2 = np.zeros((1, 2 * k)); A2[0, k:] = 1.0
        cons.append(LinearConstraint(A2, 1.0, 1.0))
        # (3) B2: ham VRAM >= gerekli
        A3 = np.zeros((1, 2 * k)); A3[0, :k] = [g.vram_gb for g in self.catalog]
        cons.append(LinearConstraint(A3, required_vram_gb, np.inf))
        # (4) güç
        if max_power_watts is not None:
            A4 = np.zeros((1, 2 * k)); A4[0, :k] = [g.power_watts for g in self.catalog]
            cons.append(LinearConstraint(A4, -np.inf, max_power_watts))
        # (5) min-tps (afin model — B4 için düğüm cezası çözüm sonrası doğrulanır)
        if min_tps is not None:
            ct = np.zeros(k); st = np.zeros(k)
            for i, g in enumerate(self.catalog):
                ct[i], st[i] = self.tp.affine_params(model, g, context_length)
            A5 = np.zeros((1, 2 * k)); A5[0, :k] = st; A5[0, k:] = ct
            cons.append(LinearConstraint(A5, min_tps, np.inf))

        integrality = np.ones(2 * k)
        lb = np.zeros(2 * k)
        per_ub = np.array([(MAX_GPUS_WITHOUT_NVLINK if not g.supports_nvlink else self.max_per_type)
                           for g in self.catalog], dtype=float)
        ub = np.concatenate([per_ub, np.ones(k)])
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub))

        # B1: infeasible ise SESSİZCE düşme — kısıtsız çöz, ama işaretle
        constraints_met = True
        warnings: List[str] = []
        if not res.success:
            constraints_met = False
            warnings.append("İstenen kısıtlar (min-TPS / max-güç) HİÇBİR konfigürasyonla "
                            "sağlanamadı. Kısıtlar gevşetilerek yalnızca VRAM'i karşılayan "
                            "en ucuz çözüm gösteriliyor.")
            res = self._solve_vram_only(model, required_vram_gb)
            if res is None or not res.success:
                return None

        x = res.x
        n_vals = np.round(x[:k]).astype(int)
        y_vals = np.round(x[k:]).astype(int)
        idx = int(np.argmax(y_vals))
        gpu = self.catalog[idx]
        n = int(n_vals[idx])
        if n <= 0:
            return None

        num_nodes = math.ceil(n / MAX_GPUS_PER_NODE)
        single = self.tp.single_stream(model, gpu, context_length)
        agg = self.tp.aggregate(model, gpu, n, single, num_nodes)

        # B4: düğüm cezası sonrası min-tps gerçekten sağlanıyor mu?
        if min_tps is not None and agg.aggregate_tps is not None and agg.aggregate_tps < min_tps - 1e-6:
            constraints_met = False
            warnings.append(f"Çok-düğüm verimlilik kaybı sonrası gerçek TPS "
                            f"({agg.aggregate_tps:.1f}) istenen min-TPS'in ({min_tps:.1f}) altında.")

        return GPUConfiguration(
            gpu=gpu, num_gpus=n, num_nodes=num_nodes, total_vram_gb=gpu.vram_gb * n,
            is_feasible=True, constraints_met=constraints_met,
            total_cost_usd=gpu.price_usd * n, estimated_tps=agg.aggregate_tps, tps_unit=agg.unit,
            solver_used="MILP (scipy/HiGHS)", warnings=warnings,
        )

    def _solve_vram_only(self, model, required_vram_gb):
        k = len(self.catalog)
        big_m = float(self.max_per_type)
        c = np.concatenate([np.array([g.price_usd for g in self.catalog]), np.zeros(k)])
        cons = []
        A1 = np.zeros((k, 2 * k))
        for i in range(k):
            A1[i, i] = 1.0; A1[i, k + i] = -big_m
        cons.append(LinearConstraint(A1, -np.inf, 0.0))
        A2 = np.zeros((1, 2 * k)); A2[0, k:] = 1.0
        cons.append(LinearConstraint(A2, 1.0, 1.0))
        A3 = np.zeros((1, 2 * k)); A3[0, :k] = [g.vram_gb for g in self.catalog]
        cons.append(LinearConstraint(A3, required_vram_gb, np.inf))
        per_ub = np.array([(MAX_GPUS_WITHOUT_NVLINK if not g.supports_nvlink else self.max_per_type)
                           for g in self.catalog], dtype=float)
        ub = np.concatenate([per_ub, np.ones(k)])
        return milp(c=c, constraints=cons, integrality=np.ones(2 * k),
                    bounds=Bounds(np.zeros(2 * k), ub))


# ==============================================================================
# OFFLOAD (B5: gerçek batch)
# ==============================================================================
@dataclass
class OffloadPlan:
    total_layers: int
    layers_on_gpu: int
    layers_on_cpu: int
    is_fully_on_gpu: bool
    effective_tps: Optional[float]
    slowdown_factor: Optional[float]
    explanation: str


class OffloadPlanner:
    def __init__(self, ram_bw=SYSTEM_RAM_BANDWIDTH_GBPS, cpu_penalty=CPU_COMPUTE_PENALTY):
        self.ram_bw = ram_bw
        self.cpu_penalty = cpu_penalty

    def plan(self, model, gpu, vram: VRAMBreakdown, full_gpu_tps, batch_size=1, num_gpus=1) -> Optional[OffloadPlan]:
        if model.num_layers is None or full_gpu_tps is None:
            return None
        avail = gpu.vram_gb * num_gpus
        reserved = vram.kv_cache_memory_gb + vram.activation_memory_gb
        for_weights = max(avail - reserved, 0.0)
        per_layer = vram.weight_memory_gb / model.num_layers
        if per_layer <= 0:
            return None
        on_gpu = min(model.num_layers, int(for_weights // per_layer))
        on_cpu = model.num_layers - on_gpu
        if on_cpu == 0:
            return OffloadPlan(model.num_layers, on_gpu, 0, True, full_gpu_tps, 1.0,
                               "Model tamamen GPU VRAM'ine sığıyor; offload gerekmiyor.")
        bpl = per_layer * BYTES_PER_GB
        t_gpu = (on_gpu * bpl) / (gpu.memory_bandwidth_gbps * BYTES_PER_GB)
        t_cpu = ((on_cpu * bpl) / (self.ram_bw * BYTES_PER_GB)) * self.cpu_penalty
        act_bytes = (model.hidden_dim or 4096) * batch_size * ACTIVATION_BYTES_PER_ELEM  # B5
        t_tr = act_bytes / (gpu.pcie_bandwidth_gbps * BYTES_PER_GB)
        t_tot = t_gpu + t_cpu + t_tr
        eff = 1.0 / t_tot if t_tot > 0 else 0.0
        slow = (t_tot * full_gpu_tps) if full_gpu_tps > 0 else None
        return OffloadPlan(model.num_layers, on_gpu, on_cpu, False, round(eff, 2),
                           round(slow, 2) if slow else None,
                           f"{on_gpu}/{model.num_layers} katman GPU'da, {on_cpu} katman RAM'de "
                           f"(~{self.cpu_penalty:.0f}× yavaş). TPS {full_gpu_tps:.1f}→{eff:.1f}.")


# ==============================================================================
# MALİYET + KARBON
# ==============================================================================
@dataclass
class CostBreakdown:
    hourly_amortization_usd: float
    hourly_energy_usd: float
    hourly_total_usd: float
    cost_per_1k_tokens_usd: Optional[float]      # C1: NaN yerine None
    emissions_g_co2_per_1k_tokens: Optional[float]


class CostModel:
    def __init__(self, hardware_lifetime_hours=DEFAULT_HARDWARE_LIFETIME_HOURS,
                 electricity_price_usd_per_kwh=0.12, carbon_intensity_g_per_kwh=None, pue=1.4):
        self.life = hardware_lifetime_hours
        self.price = electricity_price_usd_per_kwh
        self.carbon = carbon_intensity_g_per_kwh
        self.pue = pue

    def compute(self, config: GPUConfiguration, tps: Optional[float]) -> CostBreakdown:
        amort = config.total_cost_usd / self.life
        power_kw = (config.gpu.power_watts * config.num_gpus / 1000.0) * self.pue
        energy = power_kw * self.price
        hourly = amort + energy
        cost_1k = None
        emis = None
        if tps and tps > 0:
            cost_1k = (hourly / (tps * SECONDS_PER_HOUR)) * 1000
            if self.carbon is not None:
                hours_1k = (1000.0 / tps) / SECONDS_PER_HOUR
                emis = power_kw * hours_1k * self.carbon
        return CostBreakdown(round(amort, 5), round(energy, 5), round(hourly, 5),
                             round(cost_1k, 6) if cost_1k is not None else None,
                             round(emis, 4) if emis is not None else None)


# ==============================================================================
# DARBOĞAZ ANALİZİ
# ==============================================================================
@dataclass
class BottleneckAnalysis:
    bottleneck: BottleneckType
    explanation: str
    ratios: Dict[str, float] = field(default_factory=dict)


class BottleneckAnalyzer:
    def analyze(self, model, config: Optional[GPUConfiguration], vram: VRAMBreakdown,
                throughput: Optional[ThroughputResult], offload: Optional[OffloadPlan]) -> BottleneckAnalysis:
        if model.deployment == DeploymentType.CLOUD:
            return BottleneckAnalysis(BottleneckType.NETWORK_API,
                                      "Bulut modeli: asıl sınır internet hızı ve API rate-limit.")
        if offload is not None and not offload.is_fully_on_gpu:
            return BottleneckAnalysis(BottleneckType.CPU_OFFLOAD_PENALTY, offload.explanation,
                                      {"slowdown": offload.slowdown_factor or 0})
        if config is None:
            return BottleneckAnalysis(BottleneckType.VRAM,
                                      "Katalogdaki hiçbir GPU VRAM ihtiyacını karşılamıyor.",
                                      {"vram_ratio": float("inf")})
        vram_ratio = vram.total_required_gb / max(config.total_vram_gb, 1e-9)
        ratios = {"vram_ratio": round(vram_ratio, 3),
                  "power_ratio": round(config.gpu.power_watts / POWER_REFERENCE_WATTS, 3),
                  "kv_fraction": throughput.kv_fraction if throughput else 0.0}
        if model.known_bottleneck_hint is not None and vram_ratio <= 0.9:
            bt = model.known_bottleneck_hint
        elif vram_ratio > 0.9:
            bt = BottleneckType.VRAM
        elif throughput and throughput.regime == "kv-bound":
            bt = BottleneckType.KV_CACHE_BOUND
        elif config.num_gpus > 1 and not config.gpu.supports_nvlink:
            bt = BottleneckType.NVLINK_REQUIREMENT
        elif throughput and throughput.regime == "compute-bound":
            bt = BottleneckType.COMPUTE_BOUND
        else:
            bt = BottleneckType.MEMORY_BANDWIDTH
        expl = {
            BottleneckType.VRAM: f"VRAM kullanımı %{vram_ratio*100:.0f} — kapasite sınırına yakın.",
            BottleneckType.KV_CACHE_BOUND: f"Uzun bağlam: token okumasının %{ratios['kv_fraction']*100:.0f}'i KV-cache.",
            BottleneckType.NVLINK_REQUIREMENT: f"{config.num_gpus} GPU, NVLink yok — senkronizasyon kaybı.",
            BottleneckType.COMPUTE_BOUND: "Yüksek batch: işlem gücü (FLOPS) sınırlayıcı.",
            BottleneckType.MEMORY_BANDWIDTH: "Tek-istek decode: bellek bant genişliği sınırlayıcı (tipik).",
        }.get(bt, bt.value)
        return BottleneckAnalysis(bt, expl, ratios)
