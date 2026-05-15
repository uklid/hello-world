# Novel Research Gaps สำหรับ Fuzzy Linguistic Approaches ใน Clustered Federated Learning (CFL)

## TL;DR

- **Gap ที่ "publishable จริง" และยังไม่มีคนทำ** มี 4 จุดที่ defensible: (1) HFLTS-based soft cluster assignment ใน CFL, (2) 2-tuple linguistic aggregation ของ cluster decisions แบบ computing-with-words, (3) Z-numbers สำหรับ reliability-aware drift signals, และ (4) Interval Type-2 fuzzy similarity metric สำหรับ dynamic re-clustering — verification ยืนยันว่าไม่มี published work ใดในทั้ง 4 หัวข้อนี้
- **State of the art ปัจจุบัน**: FedSoft (Ruan & Joe-Wong, AAAI 2022) ทำ soft membership ในรูป numeric mixture weights, FedDrift (Jothimurugesan et al., AISTATS 2023) และ FedDAA (arXiv:2506.21054, 2025) ทำ dynamic clustering กับ drift — แต่ทั้งหมดยัง **ไม่มี linguistic semantics, ไม่มี hesitation modeling, และไม่ถ่ายทอด reliability ของ similarity signal**
- **คำแนะนำหลัก**: priority สูงสุดคือ **HFLTS-CFL** สำหรับ personalization + **Z-CFL drift detector** สำหรับ dynamic re-clustering — ทั้งสอง angle มี theoretical foundation จาก Rodriguez-Martinez-Herrera และ Zadeh, มี natural baseline (FedSoft, FedDrift), และ fit กับ IEEE TFS / Information Sciences / IEEE TCYB / AAAI

---

## Key Findings

1. **Fuzzy linguistic literature เป็น centralized ทั้งหมด** — HFLTS, 2-tuple, PLTS, Z-numbers ถูกพัฒนาในบริบท MCGDM / LSGDM ไม่เคย port ไปสู่ federated learning เลย
2. **Fuzzy ใน FL ที่มีอยู่เป็น numeric soft clustering เกือบทั้งหมด** — FFCM (Stallmann & Wilbik 2022), Federated FCM (Pedrycz 2022 IEEE TFS 30(8):3384-3388), FedFCD (Li, Lu, Pedrycz, IEEE/CAA JAS 13(3), Mar 2026), FedRFC, FedMVFCM ล้วนเป็น federated fuzzy c-means ที่ส่ง membership grade เป็น scalar ใน [0,1] — **clustering ของ data points** ไม่ใช่ของ clients
3. **Soft membership ใน CFL มีคนทำแล้ว แต่ไม่มีใครทำ linguistic** — FedSoft (AAAI 2022) ใช้ mixture weights, FedPrism (Kumbhakar, Srivastava & Lone, arXiv:2603.08252, 2026), FedPLC (Sensors 26(1):283, 2026) เป็น numeric — ยังไม่มี "client X เป็นสมาชิก cluster A ระดับ very high, cluster B ระดับ medium" แบบ linguistic terms
4. **Federated drift detection มีเยอะแต่ไม่มี linguistic severity grading** — FedDrift, FedDAA, FIELDING (Li, Avdiukhin, Shahout, Ivkin, Braverman & Yu, arXiv:2411.01580, Nov 2024), FedPLC ใช้ numeric thresholds — ยังไม่มี linguistic rule-based decisions เช่น IF drift_severity IS moderate AND confidence IS high THEN re-cluster
5. **Type-2 fuzzy + FL ยัง virgin territory** — FedFNN (Zhang, Shi, Chang, Lin, IEEE TFS, arXiv:2210.14393) ใช้ Type-1 fuzzy rules เท่านั้น ไม่มี published work ที่ใช้ IT2/GT2 ใน CFL similarity metric

---

## Research Gaps ที่ Defensible และยังไม่มีคนทำ

### Gap A — HFLTS-based Soft Cluster Assignment ใน CFL (priority สูงสุดสำหรับ personalization)

**Problem statement**: ใน soft CFL ปัจจุบัน (FedSoft, FedPrism) client X ถูก assign mixture weights เป็น vector เช่น [0.4, 0.35, 0.25] ซึ่งตีความยากและไม่ capture hesitation เมื่อ client อยู่ใกล้กึ่งกลางระหว่าง 2 clusters

**Novel idea**: ใช้ HFLTS เก็บ membership เป็น comparative expression เช่น "client X belongs to cluster A at least high" หรือ "between medium and high" — server aggregate ผ่าน envelope operations + HFLTS distance/similarity measures ของ Liao, Xu & Zeng (*Information Sciences* 271:125-142, 2014)

**Theoretical foundation**: Rodriguez-Martinez-Herrera (IEEE TFS 20(1):109-119, 2012) + Liao-Xu-Zeng distance measures + Wei-Zhao-Tang HFLTS operators (*IEEE TFS* 22(3):575-585, 2014)

**Baseline**: FedSoft (AAAI 2022), IFCA (NeurIPS 2020), Sattler CFL (TNNLS 2021), FedPrism (arXiv:2603.08252, 2026)

**Metrics**: cluster accuracy (ARI, NMI), personalization accuracy (per-client test acc), interpretability score (number of comparative expressions per client), communication cost

**Publishable venue**: IEEE TFS, Information Sciences, IEEE TCYB

**Benchmark threshold for go/no-go**: ถ้า HFLTS-CFL ไม่สามารถ match FedSoft accuracy ภายใน 2% บน CIFAR-10 non-IID → pivot ไป interpretability-only angle

---

### Gap B — 2-tuple Linguistic Aggregation of Cluster Decisions (computing-with-words angle)

**Problem statement**: ใน CFL ที่ใช้ multiple similarity signals (cosine similarity, loss-based, classifier-head similarity ที่ FedPLC ใช้) การ aggregate เป็น hard decision ทำให้สูญเสีย information

**Novel idea**: encode each similarity signal เป็น linguistic 2-tuple (s_i, alpha), aggregate ด้วย 2-tuple LOWA operator, แล้ว defuzzify เป็น cluster assignment — output มาพร้อม symbolic translation ที่ตีความได้

**Theoretical foundation**: Herrera-Martinez (*IEEE TFS* 8(6):746-752, 2000) + multigranular 2-tuple (Herrera-Martinez *TSMCB* 31(2):227-234, 2001)

**Baseline**: FIELDING (Li, Avdiukhin, Shahout, Ivkin, Braverman & Yu, arXiv:2411.01580, 2024)

**Publishable venue**: IEEE TFS, Information Fusion

---

### Gap C — Z-numbers สำหรับ Reliability-Aware Drift Detection (priority สูงสุดสำหรับ drift)

**Problem statement**: FedDrift, FedDAA, FIELDING ตรวจ drift ผ่าน loss/gradient signal เดียว — แต่ใน FL signal เหล่านี้ noisy เพราะ (a) limited local data, (b) stale updates, (c) Byzantine clients. การ trigger re-clustering บน signal ที่ confidence ต่ำทำให้ over-react

**Novel idea**: ทุก drift signal บน client ส่งเป็น Z-number Z = (A, B) — A = fuzzy magnitude ของ drift, B = reliability. Server combine Z-numbers ผ่าน Z-number aggregation แล้วใช้ fuzzy rule

**Theoretical foundation**: Zadeh (*Information Sciences* 181(14):2923-2932, 2011) + Z-number aggregation theory (Yager 2012, Aliev et al. 2016)

**Baseline**: FedDrift, FedDAA, FedPLC

**Publishable venue**: IEEE TFS, Information Sciences, IEEE TKDE, AAAI

**Benchmark threshold for go/no-go**: ถ้า Z-CFL ไม่ลด false re-clustering rate ลง ≥30% เทียบกับ FedDrift บน synthetic drift → pivot ไป IT2 (Gap D)

---

### Gap D — Interval Type-2 Fuzzy Similarity Metric สำหรับ CFL (uncertainty-of-similarity angle)

**Problem statement**: cosine similarity ของ gradients (Sattler CFL, TNNLS 2021) หรือ loss-based identity (IFCA, NeurIPS 2020) เป็น noisy estimators ของ true client similarity. ปัจจุบันใช้ point estimate

**Novel idea**: IT2 fuzzy similarity — แทน similarity ด้วย footprint of uncertainty (FOU) จาก variance ของ similarity across training rounds — cluster assignment ทำผ่าน type reduction

**Publishable venue**: IEEE TFS, IEEE TCYB

---

### Gap E — Linguistic Drift-Severity Rules + Granular Re-clustering (extension ของ C)

**Idea**: granular computing — drift detection ที่ multiple resolutions (per-feature, per-class, per-client) แล้วใช้ HFLTS rules-based system ตัดสินใจ split/merge clusters

**Foundation**: Pedrycz "Advancing Federated Learning with Granular Computing" (Fuzzy Inf. Eng. 15(1):1-13, 2023)

---

### Gap F — Linguistic Cluster Descriptions (interpretability layer)

**Idea**: หลัง CFL converge ให้สร้าง linguistic summary ของแต่ละ cluster เช่น "Cluster A = clients with high data volume, low label noise, slow communication"

**Foundation**: Kaczmarek-Majer et al. "Plenary: explaining black-box models in natural language through fuzzy linguistic summaries" (*Information Sciences* 2022)

---

## Recommended Path สำหรับ Paper แรก

1. **เริ่มต้น**: Gap A (HFLTS-CFL) — ภายใน 6 เดือน ได้ paper TFS ที่ defensible
   - implement บน FedSoft codebase (AAAI 2022)
   - แทน mixture weight scalar ด้วย HFLTS envelope
   - context-free grammar G_H = {high, medium, low, very, between, at_least}
   - Dataset: CIFAR-10/100 non-IID, FEMNIST, Shakespeare
   - Metrics: per-client accuracy, ARI vs ground truth, interpretability

2. **Follow-up**: Gap C (Z-CFL drift) — บน FedDrift codebase (AISTATS 2023)

3. **Long-term**: รวม A+C เป็น "Linguistic CFL Framework" หรือ TFS journal version

---

## Caveats

1. **Verification ใช้ ~7 targeted queries** — confidence สูงในการ confirm absence แต่ negative claims ใน paper ควรเขียนเป็น "to the best of our knowledge no published work applies HFLTS to CFL" และ supplement ด้วย Scopus systematic search
2. **FedFCD (Li, Lu, Pedrycz, IEEE/CAA JAS 13(3), Mar 2026)** เป็น in-press — ติดตาม final version
3. **Pedrycz "Federated FCM" letter (IEEE TFS 30(8):3384-3388, 2022)** เป็น short paper — ไม่เพียงพอที่จะถือว่า federated fuzzy clustering ถูก solve แล้ว
4. **Type-2 fuzzy in FL** — ตรวจ FUZZ-IEEE 2021-2024 proceedings เพิ่มเติม
5. **Granular computing + FL** มี Pedrycz 2023 manifesto + GCM-FL (Springer 2024) แล้ว — Gap E ต้อง position ชัดว่าต่างจาก GCM-FL อย่างไร
6. **HFLTS aggregation operators (HFLWA, HFLOWA)** มี subtle issues ในเรื่อง ordering ของ envelopes — แนะนำใช้ Liao-Xu-Zeng 2014 distance measures ที่ proven properties แล้ว
7. **Rjoub et al. "Trust-Augmented" (Inf. Syst. Frontiers 2024)** ใช้ "Z-score" statistical (ไม่ใช่ Zadeh Z-numbers)
