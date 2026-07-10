#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FİZİBİLİTE KATMANI (feasibility.py)
================================================================================
engine.py'deki düzeltilmiş hesap motorunu kullanarak bir yatırım/geri-ödeme
(ROI) analizi üretir. Cevapladığı sorular:
  • Hangi model + hangi donanım?           → engine (MILP)
  • Ne kadar VRAM / hangi hız?              → engine (VRAM + throughput)
  • Ne kadar YATIRIM (capex + sunucu)?      → burada
  • Aylık işletme maliyeti (opex)?          → burada
  • Bulut'a göre ne kadar tasarruf?         → burada
  • Kaç ayda kendini öder (payback)?        → burada
  • 36 ayda ROI ve kümülatif eğri?          → burada
Tüm para değerleri USD hesaplanır, istenirse TRY kuruyla da döndürülür.
================================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

from engine import (
    AIModelSpec, GPUSpec, DeploymentType, ArchitectureType,
    VRAMCalculator, ThroughputEstimator, MILPGPUSelector, OffloadPlanner,
    CostModel, BottleneckAnalyzer, HOURS_PER_MONTH, SECONDS_PER_HOUR,
    DEFAULT_HARDWARE_LIFETIME_HOURS,
)


# Donanımın ekonomik ömrü. Geri ödeme bunu aşıyorsa yatırım önerilmez:
# yatırım kendini çıkarmadan donanım eskir/değersizleşir.
HARDWARE_LIFE_MONTHS: int = 36
MAX_PAYBACK_MONTHS: int = 240   # iskontolu geri ödeme aramasında üst sınır


@dataclass
class FeasibilityInputs:
    model_name: str
    context_length: Optional[int] = None
    batch_size: int = 8
    monthly_volume: float = 50_000_000       # aylık token (veya diffusion'da görsel) hacmi
    # Temmuz 2026 sanayi elektriği ≈ 5,4 TL/kWh ≈ 0.115 USD/kWh
    electricity_price_usd_per_kwh: float = 0.115
    pue: float = 1.4
    usd_try: float = 46.9                     # 1 USD = ? TL (Temmuz 2026)
    # Yıllık sermaye maliyeti (USD bazlı fırsat/finansman maliyeti). Capex'in
    # bağlanmasının bedeli: paranın zaman değeri buradan modele girer.
    annual_capital_cost_pct: float = 15.0
    server_overhead_pct: float = 0.35         # şasi, ağ, kurulum, soğutma ek maliyeti
    fixed_monthly_ops_usd: float = 0.0        # bakım/personel/kolokasyon (opsiyonel)
    cloud_rate_usd_per_1k: Optional[float] = None   # bulut alternatifi (yoksa modelin katalog değeri)
    min_tps: Optional[float] = None
    max_power_watts: Optional[float] = None
    carbon_intensity_g_per_kwh: Optional[float] = None
    horizon_months: int = 36
    always_on: bool = False                   # True: 7/24 enerji; False: yalnız işlenen hacmin enerjisi


def _try(v: Optional[float], rate: float) -> Optional[float]:
    return None if v is None else round(v * rate, 2)


class FeasibilityAnalyzer:
    def __init__(self, gpu_catalog: List[GPUSpec], model_catalog: List[AIModelSpec]):
        self.gpus = gpu_catalog
        self.models = model_catalog
        self.vram = VRAMCalculator()
        self.tp = ThroughputEstimator()
        self.milp = MILPGPUSelector(gpu_catalog, self.tp)
        self.offload = OffloadPlanner()
        self.bottleneck = BottleneckAnalyzer()

    def _find_model(self, name: str) -> Optional[AIModelSpec]:
        from engine import find_model_by_name
        return find_model_by_name(name, self.models)

    def analyze(self, inp: FeasibilityInputs) -> Dict[str, Any]:
        model = self._find_model(inp.model_name)
        if model is None:
            return {"error": f"Model bulunamadı: '{inp.model_name}'"}

        rate = inp.usd_try
        cost_model = CostModel(
            electricity_price_usd_per_kwh=inp.electricity_price_usd_per_kwh,
            carbon_intensity_g_per_kwh=inp.carbon_intensity_g_per_kwh,
            pue=inp.pue,
        )

        # -------- BULUT MODELİ: yerel donanım yok, sadece bulut maliyeti --------
        if model.deployment == DeploymentType.CLOUD:
            cloud_rate = inp.cloud_rate_usd_per_1k or model.cloud_token_cost_usd_per_1k or 0.01
            monthly_cloud = cloud_rate * (inp.monthly_volume / 1000.0)
            return {
                "model": self._model_dict(model),
                "is_cloud": True,
                "recommendation": "Bu bir bulut/kapalı model — yerel donanım yatırımı uygulanamaz.",
                "cloud_monthly_usd": round(monthly_cloud, 2),
                "cloud_monthly_try": _try(monthly_cloud, rate),
                "cloud_rate_usd_per_1k": cloud_rate,
                "unit": "token",
                "warnings": ["Yerel dağıtım için açık-ağırlıklı (LOCAL) bir model seçin."],
            }

        # -------- YEREL MODEL: tam fizibilite --------
        vram = self.vram.total_required(model, inp.context_length, inp.batch_size)
        config = self.milp.solve(model, vram.total_required_gb, inp.context_length,
                                 min_tps=inp.min_tps, max_power_watts=inp.max_power_watts)

        offload_plan = None
        single = self.tp.single_stream(model, self.gpus[0], inp.context_length, inp.batch_size)
        unit = single.unit  # token/s veya image/s

        if config is None:
            # Hiçbir çözüm yok → en büyük tek GPU ile offload dene
            biggest = max(self.gpus, key=lambda g: g.vram_gb)
            full = self.tp.single_stream(model, biggest, inp.context_length, inp.batch_size)
            offload_plan = self.offload.plan(model, biggest, vram, full.single_stream_tps,
                                             inp.batch_size, num_gpus=1)

        # Aggregate throughput (config varsa)
        agg_tps = None
        if config is not None:
            s = self.tp.single_stream(model, config.gpu, inp.context_length, inp.batch_size)
            agg = self.tp.aggregate(model, config.gpu, config.num_gpus, s, config.num_nodes)
            agg_tps = agg.aggregate_tps
            unit = agg.unit
        elif offload_plan is not None:
            agg_tps = offload_plan.effective_tps

        cost = cost_model.compute(config, agg_tps) if config is not None else None
        bottleneck = self.bottleneck.analyze(model, config, vram, single, offload_plan)

        # ---------------- YATIRIM & GERİ ÖDEME ----------------
        roi = None
        capacity_per_month = None
        if config is not None and agg_tps and agg_tps > 0:
            capacity_per_month = agg_tps * HOURS_PER_MONTH * SECONDS_PER_HOUR  # birim/ay (7/24 tam yük)
            roi = self._roi(inp, config, cost, agg_tps, capacity_per_month, unit, model, rate)

        return {
            "model": self._model_dict(model),
            "is_cloud": False,
            "unit": unit,
            "vram": {"breakdown": vram.as_dict(), "trace": vram.formula_trace},
            "config": self._config_dict(config, rate) if config else None,
            "throughput": {
                "single_stream": single.single_stream_tps,
                "aggregate": agg_tps,
                "unit": unit,
                "regime": single.regime,
                "kv_fraction": single.kv_fraction,
                "explanation": single.explanation,
            },
            "cost": self._cost_dict(cost, rate) if cost else None,
            "offload": asdict(offload_plan) if offload_plan else None,
            "bottleneck": {"type": bottleneck.bottleneck.value, "explanation": bottleneck.explanation,
                           "ratios": bottleneck.ratios},
            "capacity_per_month": round(capacity_per_month) if capacity_per_month else None,
            "roi": roi,
            "warnings": (config.warnings if config else
                         ["Katalogdaki tek GPU'lar bile yetmiyor; yalnızca offload senaryosu mümkün."]),
        }

    # ---------------------------------------------------------------- ROI
    def _roi(self, inp: FeasibilityInputs, config, cost, agg_tps, capacity, unit,
             model: AIModelSpec, rate: float) -> Dict[str, Any]:
        # Yatırım (capex)
        hardware_usd = config.total_cost_usd
        upfront_usd = hardware_usd * (1 + inp.server_overhead_pct)

        # Aylık opex (capex hariç): enerji + sabit ops
        power_kw = (config.gpu.power_watts * config.num_gpus / 1000.0) * inp.pue
        if inp.always_on:
            energy_hours = HOURS_PER_MONTH
        else:
            energy_hours = min(inp.monthly_volume / agg_tps / SECONDS_PER_HOUR, HOURS_PER_MONTH)
        energy_month = power_kw * energy_hours * inp.electricity_price_usd_per_kwh
        opex_month = energy_month + inp.fixed_monthly_ops_usd

        # Kapasite yeterli mi?
        utilization = inp.monthly_volume / capacity if capacity > 0 else float("inf")
        capacity_ok = utilization <= 1.0

        # Bulut alternatifi
        cloud_rate = inp.cloud_rate_usd_per_1k
        if cloud_rate is None:
            cloud_rate = model.cloud_token_cost_usd_per_1k or _default_cloud_rate(unit)
        cloud_month = cloud_rate * (inp.monthly_volume / 1000.0) if unit == "token/s" \
            else cloud_rate * inp.monthly_volume  # image başına
        cloud_unit_label = "USD/1K token" if unit == "token/s" else "USD/görsel"

        # ---------------- PARANIN ZAMAN DEĞERİ ----------------
        # CoT: Capex bedava değildir — ya kredi faizi ödenir ya da o para başka
        # bir yerde getiri sağlardı. Bunu modele katmadan yapılan geri ödeme
        # hesabı, "190 ayda döner" gibi gerçekte ASLA dönmeyecek yatırımları
        # makul gösterir. Bu yüzden hem finansman maliyetini hem de İSKONTOLU
        # geri ödemeyi hesaplıyoruz.
        r = max(inp.annual_capital_cost_pct, 0.0) / 100.0     # yıllık sermaye maliyeti
        i = (1 + r) ** (1 / 12.0) - 1 if r > 0 else 0.0       # aylık oran
        N = inp.horizon_months
        financing_cost = upfront_usd * ((1 + r) ** (N / 12.0) - 1) if r > 0 else 0.0

        monthly_savings = cloud_month - opex_month
        nominal_payback = upfront_usd / monthly_savings if monthly_savings > 0 else None

        # İskontolu geri ödeme: tasarrufların BUGÜNKÜ değeri capex'i ne zaman karşılar?
        # i > 0 iken iskontolu tasarruf toplamı en fazla (aylık tasarruf / i) olabilir.
        # Capex bu tavanın üstündeyse yatırım matematiksel olarak ASLA kendini ödemez.
        discounted_payback: Optional[float] = None
        never_pays_back = False
        if monthly_savings > 0:
            if i <= 0:
                discounted_payback = nominal_payback
            elif (monthly_savings / i) <= upfront_usd:
                never_pays_back = True     # iskontolu tavan capex'in altında
            else:
                acc = 0.0
                for m in range(1, MAX_PAYBACK_MONTHS + 1):
                    acc += monthly_savings / ((1 + i) ** m)
                    if acc >= upfront_usd:
                        discounted_payback = float(m)
                        break
        else:
            never_pays_back = True

        payback_months = discounted_payback
        # Yatırım tavsiyesi: donanım ömrü (36 ay) içinde kendini ödemeli.
        invest_recommended = bool(capacity_ok and payback_months is not None
                                  and payback_months <= HARDWARE_LIFE_MONTHS)

        # Amortisman bazlı aylık yerel toplam (amort + opex) — TCO kıyası
        amort_month = (cost.hourly_amortization_usd * HOURS_PER_MONTH) if cost else \
            (hardware_usd / DEFAULT_HARDWARE_LIFETIME_HOURS * HOURS_PER_MONTH)
        local_month_tco = amort_month + opex_month

        # Ufuk boyunca kümülatif eğri (grafik için) — yerel tarafa finansman maliyeti dahil
        months = list(range(0, inp.horizon_months + 1))
        cum_local = [round(upfront_usd + opex_month * m
                           + (upfront_usd * ((1 + r) ** (m / 12.0) - 1) if r > 0 else 0.0), 2)
                     for m in months]
        cum_cloud = [round(cloud_month * m, 2) for m in months]

        net_savings_horizon = cum_cloud[-1] - cum_local[-1]
        roi_pct = (net_savings_horizon / upfront_usd * 100.0) if upfront_usd > 0 else None

        # Break-even AYLIK HACİM: local_tco(V)=cloud(V)
        # amort_month + fixed + energy_rate*V = cloud_rate_per_unit*V
        energy_rate_per_unit = (opex_month - inp.fixed_monthly_ops_usd) / max(inp.monthly_volume, 1)
        cloud_rate_per_unit = (cloud_rate / 1000.0) if unit == "token/s" else cloud_rate
        denom = cloud_rate_per_unit - energy_rate_per_unit
        break_even_volume = ((amort_month + inp.fixed_monthly_ops_usd) / denom) if denom > 0 else None

        return {
            "hardware_usd": round(hardware_usd, 2),
            "hardware_try": _try(hardware_usd, rate),
            "server_overhead_pct": inp.server_overhead_pct,
            "upfront_investment_usd": round(upfront_usd, 2),
            "upfront_investment_try": _try(upfront_usd, rate),
            "opex_month_usd": round(opex_month, 2),
            "opex_month_try": _try(opex_month, rate),
            "energy_month_usd": round(energy_month, 2),
            "local_month_tco_usd": round(local_month_tco, 2),
            "local_month_tco_try": _try(local_month_tco, rate),
            "cloud_rate_used": cloud_rate,
            "cloud_rate_unit": cloud_unit_label,
            "cloud_month_usd": round(cloud_month, 2),
            "cloud_month_try": _try(cloud_month, rate),
            "monthly_savings_usd": round(monthly_savings, 2),
            "monthly_savings_try": _try(monthly_savings, rate),
            # Paranın zaman değeri
            "annual_capital_cost_pct": inp.annual_capital_cost_pct,
            "financing_cost_usd": round(financing_cost, 2),
            "financing_cost_try": _try(financing_cost, rate),
            "nominal_payback_months": round(nominal_payback, 1) if nominal_payback else None,
            "payback_months": round(payback_months, 1) if payback_months else None,
            "never_pays_back": never_pays_back,
            "invest_recommended": invest_recommended,
            "hardware_life_months": HARDWARE_LIFE_MONTHS,
            "payback_verdict": _payback_verdict(payback_months, capacity_ok, never_pays_back),
            "utilization": round(utilization, 3),
            "capacity_ok": capacity_ok,
            "break_even_volume_per_month": round(break_even_volume) if break_even_volume else None,
            "horizon_months": inp.horizon_months,
            "roi_pct_horizon": round(roi_pct, 1) if roi_pct is not None else None,
            "net_savings_horizon_usd": round(net_savings_horizon, 2),
            "net_savings_horizon_try": _try(net_savings_horizon, rate),
            "chart": {"months": months, "cumulative_local_usd": cum_local, "cumulative_cloud_usd": cum_cloud},
        }

    # ---------------------------------------------------------------- serializers
    def _model_dict(self, m: AIModelSpec) -> Dict[str, Any]:
        return {
            "name": m.name, "use_case": m.use_case, "deployment": m.deployment.name,
            "architecture": m.architecture.value, "precision": m.quant().label,
            "params_billions": m.params_billions,
            "active_params_billions": m.effective_active_params_billions(),
            "context_length": m.default_context_length, "notes": m.notes,
        }

    def _config_dict(self, c, rate) -> Dict[str, Any]:
        return {
            "label": c.label(), "gpu": c.gpu.name, "num_gpus": c.num_gpus, "num_nodes": c.num_nodes,
            "total_vram_gb": c.total_vram_gb, "hardware_cost_usd": round(c.total_cost_usd, 2),
            "hardware_cost_try": _try(c.total_cost_usd, rate),
            "estimated_tps": c.estimated_tps, "tps_unit": c.tps_unit,
            "constraints_met": c.constraints_met, "solver": c.solver_used, "warnings": c.warnings,
        }

    def _cost_dict(self, cost, rate) -> Dict[str, Any]:
        return {
            "hourly_total_usd": cost.hourly_total_usd,
            "hourly_amortization_usd": cost.hourly_amortization_usd,
            "hourly_energy_usd": cost.hourly_energy_usd,
            "cost_per_1k_tokens_usd": cost.cost_per_1k_tokens_usd,
            "cost_per_1k_tokens_try": _try(cost.cost_per_1k_tokens_usd, rate),
            "emissions_g_co2_per_1k_tokens": cost.emissions_g_co2_per_1k_tokens,
        }


def _default_cloud_rate(unit: str) -> float:
    # Kaba referans: token modelleri için ~$0.01/1K çıktı token; diffusion için ~$0.04/görsel
    return 0.01 if unit == "token/s" else 0.04


def _payback_verdict(payback_months: Optional[float], capacity_ok: bool,
                     never_pays_back: bool = False) -> str:
    if not capacity_ok:
        return "Seçilen donanım bu aylık hacmi 7/24 bile karşılayamıyor — daha fazla/güçlü GPU gerekir."
    if never_pays_back or payback_months is None:
        return ("Bu hacimde yatırım kendini ÖDEMİYOR. Sermaye maliyeti hesaba katıldığında "
                "tasarruflar capex'i hiçbir zaman karşılamıyor — şimdilik bulut/abonelik doğru tercih.")
    if payback_months <= 12:
        return f"Çok cazip: yatırım ~{payback_months:.0f} ayda geri döner (sermaye maliyeti dahil)."
    if payback_months <= HARDWARE_LIFE_MONTHS:
        return f"Makul: yatırım ~{payback_months:.0f} ayda, donanım ömrü ({HARDWARE_LIFE_MONTHS} ay) içinde geri döner."
    return (f"Önerilmez: geri ödeme ~{payback_months:.0f} ay, donanım ömrünü "
            f"({HARDWARE_LIFE_MONTHS} ay) aşıyor — yatırım kendini çıkarmadan donanım eskiyor.")
