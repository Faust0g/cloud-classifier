# Evaluación final

La prueba final se ejecutó una única vez con el artefacto de calibración existente.

## Test

- Fotos: 411
- Exactitud top-1: 51.1% (46.3%–55.9%)
- Exactitud top-2: 69.8% (65.2%–74.1%)
- Precisión de resultados firmes: 59.6% (51.8%–67.0%)
- Cobertura del par: 16.5% (13.3%–20.4%); entre no firmes nube: 36.6% (30.0%–43.7%)

| Resultado | Cantidad | Proporción (IC Wilson 95%) |
|---|---:|---:|
| firme | 156 | 38.0% (33.4%–42.7%) |
| par | 68 | 16.5% (13.3%–20.4%) |
| repetir | 118 | 28.7% (24.5%–33.3%) |
| no_nube | 69 | 16.8% (13.5%–20.7%) |

### Por clase real

| Clase | Cantidad | Top-1 correctas |
|---|---:|---:|
| altocumulus | 30 | 11 |
| altostratus | 27 | 8 |
| cirrocumulus | 40 | 17 |
| cirrostratus | 42 | 14 |
| cirrus | 22 | 8 |
| cumulonimbus | 35 | 23 |
| cumulus | 26 | 11 |
| nimbostratus | 37 | 18 |
| no_nube | 75 | 68 |
| stratocumulus | 49 | 28 |
| stratus | 28 | 4 |

### Por fuente

| Fuente | Cantidad | Top-1 | Top-2 | Firmes | Pares | Repetir | No nube | Precisión firmes | Cobertura par |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ccsn | 340 | 154 (45.3%) | 224 (65.9%) | 140 | 66 | 107 | 27 | 59.3% (51.0%–67.1%) | 19.4% |
| open_images | 44 | 42 (95.5%) | 43 (97.7%) | 0 | 0 | 2 | 42 | n/a | 0.0% |
| sociedad | 27 | 14 (51.9%) | 20 (74.1%) | 16 | 2 | 9 | 0 | 62.5% (38.6%–81.5%) | 7.4% |

## Fotos propias

- Fotos: 122
- Sesiones distintas: 1
- Exactitud top-1: 36.9% (28.8%–45.7%)
- Exactitud top-2: 51.6% (42.9%–60.3%)
- Precisión de resultados firmes: 60.5% (45.6%–73.6%)
- Cobertura del par: 17.2% (11.5%–24.9%); entre no firmes nube: 30.0% (20.5%–41.5%)

| Resultado | Cantidad | Proporción (IC Wilson 95%) |
|---|---:|---:|
| firme | 43 | 35.2% (27.3%–44.1%) |
| par | 21 | 17.2% (11.5%–24.9%) |
| repetir | 49 | 40.2% (31.9%–49.0%) |
| no_nube | 9 | 7.4% (3.9%–13.4%) |

### Por clase real

| Clase | Cantidad | Top-1 correctas |
|---|---:|---:|
| altocumulus | 6 | 1 |
| altostratus | 6 | 1 |
| cirrocumulus | 6 | 4 |
| cirrostratus | 7 | 5 |
| cirrus | 16 | 10 |
| cumulonimbus | 14 | 5 |
| cumulus | 24 | 9 |
| nimbostratus | 7 | 4 |
| no_nube | 5 | 5 |
| stratocumulus | 14 | 0 |
| stratus | 17 | 1 |

### Por fuente

| Fuente | Cantidad | Top-1 | Top-2 | Firmes | Pares | Repetir | No nube | Precisión firmes | Cobertura par |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| propias | 122 | 45 (36.9%) | 63 (51.6%) | 43 | 21 | 49 | 9 | 60.5% (45.6%–73.6%) | 17.2% |

## Doble etiqueta (descriptivo; excluido de las métricas)

| Grupo | Resultado | Clases top-1 / top-2 |
|---|---|---|
| group_017067bf3749a5c3 | repetir | cirrocumulus, altocumulus |
| group_02f9fb0bcba7681b | firme | cumulonimbus, cumulus |
| group_033faaacc84b69ce | firme | cumulus, altocumulus |
| group_08bcaaf44f8f3759 | repetir | nimbostratus, altostratus |
| group_09280bf4f7a4a4a4 | firme | stratocumulus, stratus |
| group_0a4786f549550803 | firme | altocumulus, cirrocumulus |
| group_0aba1ed53a3f3a6b | repetir | nimbostratus, stratus |
| group_0f0ece4800c4feaa | par | cumulus, cirrostratus |
| group_102889cbe554e0e2 | firme | stratocumulus, stratus |
| group_14397e38d728b29a | par | stratocumulus, stratus |
| group_149fe7cf069d1cb5 | repetir | cirrostratus, cirrus |
| group_15f3c8325cb6d495 | par | cirrostratus, stratocumulus |
| group_160c975aa60f7a74 | par | cumulonimbus, cirrostratus |
| group_1a3c8fa65579c4b5 | repetir | altocumulus, altostratus |
| group_1b7a6db51116aab6 | firme | stratus, stratocumulus |
| group_1bcab97809b1f7b0 | par | stratus, stratocumulus |
| group_1d3710ded5346bf1 | firme | nimbostratus, stratocumulus |
| group_273d79dce80e8df0 | par | stratocumulus, stratus |
| group_29baef8a008a9b44 | firme | cirrus, cirrostratus |
| group_2b115154560e3ed9 | firme | cumulonimbus, cumulus |
| group_2ed891626dc832e8 | firme | stratocumulus, nimbostratus |
| group_2f8a600812c0224c | repetir | nimbostratus, no_nube |
| group_31db20258986fbfa | firme | altostratus, altocumulus |
| group_381f924b8c3b4a97 | repetir | cirrus, stratus |
| group_3904c46e4a788ba2 | firme | cumulonimbus, cumulus |
| group_39681a2f4831b6b8 | firme | altostratus, no_nube |
| group_3acd20b757c9f10f | firme | stratocumulus, stratus |
| group_3bf9b8a37c567462 | firme | cumulonimbus, altocumulus |
| group_3f2c53f9bfadd75b | firme | cirrostratus, stratus |
| group_40a959b0b03ac2d8 | repetir | cirrostratus, altostratus |
| group_42e94d4c1524a0e7 | firme | stratocumulus, stratus |
| group_450fa340d8a7b9fb | firme | stratocumulus, stratus |
| group_47d4a1fd22c7ee4a | repetir | altocumulus, stratocumulus |
| group_5345188c3e42aa92 | firme | cirrocumulus, altocumulus |
| group_54bc196fb368d217 | repetir | altocumulus, stratocumulus |
| group_57b2c18795e78346 | repetir | stratocumulus, stratus |
| group_58f13bf323415b40 | repetir | stratus, nimbostratus |
| group_5cb418bb727095b8 | firme | cumulus, stratus |
| group_60ba6473d79c73f4 | firme | altocumulus, cirrocumulus |
| group_618eb50a385231df | firme | cumulus, cumulonimbus |
| group_61bbc8c9041639ef | par | cirrus, cirrostratus |
| group_6492735d1efe812f | firme | nimbostratus, stratus |
| group_64fe896b3fd6e79b | repetir | stratocumulus, nimbostratus |
| group_698977b1a01ddd40 | repetir | nimbostratus, stratocumulus |
| group_6be40d205a61dad7 | par | cirrostratus, cirrus |
| group_6c5c91d72930e32d | par | stratus, stratocumulus |
| group_6e81c6c067ce8b5e | repetir | stratus, nimbostratus |
| group_7165475fbfb7b4c4 | firme | cumulus, cirrocumulus |
| group_76e8dbf312725a58 | firme | cumulonimbus, cumulus |
| group_791acdc69541d9d2 | par | stratocumulus, nimbostratus |
| group_7ab8fa360a791bd4 | par | cumulonimbus, altocumulus |
| group_7aba60851cec65b0 | repetir | cumulonimbus, altocumulus |
| group_7bf9c85cce7d2746 | firme | altostratus, stratocumulus |
| group_7f65695d2caf0723 | repetir | stratus, altostratus |
| group_7ff5f8753511fc1d | firme | nimbostratus, stratocumulus |
| group_811aa2c58e43a635 | firme | stratocumulus, cirrocumulus |
| group_8342dd30b903fa34 | par | nimbostratus, stratus |
| group_846dc7deb8c4f985 | firme | stratocumulus, stratus |
| group_852a3feb214b7e99 | firme | cirrus, cirrostratus |
| group_88d7480ec497f58c | repetir | cumulonimbus, altostratus |
| group_8cc92bffbb42664d | firme | nimbostratus, stratocumulus |
| group_8e611bb3cb9c221e | par | cumulonimbus, cumulus |
| group_8ed411b42993d8d1 | par | stratocumulus, stratus |
| group_919b9f93ba792541 | repetir | cirrostratus, altocumulus |
| group_9417679c4c27379f | no_nube | no_nube, altocumulus |
| group_95aea0883b792c53 | firme | altocumulus, nimbostratus |
| group_9b31900177eb66bf | par | cirrostratus, cirrocumulus |
| group_9c5a2a78574674ee | par | stratocumulus, nimbostratus |
| group_9d3b40307372cd5a | par | stratocumulus, cumulonimbus |
| group_9d5daecae6447175 | par | stratocumulus, stratus |
| group_a2d65b2690d4e138 | firme | stratocumulus, altocumulus |
| group_a39ca76f9fbdbce2 | repetir | stratus, nimbostratus |
| group_a5168942f417353e | firme | cirrus, cirrostratus |
| group_a522a7ba83f376df | firme | stratocumulus, nimbostratus |
| group_a52ab9899590b02b | par | cirrostratus, altocumulus |
| group_a55d1f681b89bf1f | repetir | stratocumulus, cumulonimbus |
| group_a5cd992de72e5e69 | par | cirrocumulus, altocumulus |
| group_a70bb1e238b1f71b | par | nimbostratus, cumulus |
| group_a82816da1458af3a | firme | nimbostratus, stratus |
| group_a8bc881a2a7d69ac | repetir | stratocumulus, nimbostratus |
| group_aa3204e241ad8ed0 | repetir | cirrostratus, cumulus |
| group_aec5ac9cbae86c11 | firme | nimbostratus, altocumulus |
| group_af19e826f29a5e78 | firme | cirrocumulus, altocumulus |
| group_af5a256daca42288 | firme | nimbostratus, stratocumulus |
| group_b08ddc0e2958868f | par | stratus, nimbostratus |
| group_b1d626429e91c7be | firme | nimbostratus, stratocumulus |
| group_b4dd87dcb0938fd4 | firme | stratocumulus, stratus |
| group_b6c459d89a482250 | firme | stratus, cirrus |
| group_ba7d884512ab4b88 | firme | stratus, stratocumulus |
| group_bb47f5f8f9373071 | repetir | stratocumulus, cirrus |
| group_bf6b0117d6fadae7 | repetir | stratocumulus, cirrostratus |
| group_bfe6362f14e5f322 | firme | cirrus, cirrostratus |
| group_c2e7f66807437427 | par | stratocumulus, stratus |
| group_c541e1cdb0f293cc | par | cirrostratus, cumulonimbus |
| group_c73a2dbbd8533d5e | firme | cumulus, altocumulus |
| group_c7995c7beb77b465 | repetir | stratus, cirrostratus |
| group_c90a75eafa9dcde9 | firme | nimbostratus, stratocumulus |
| group_ca32a654fdc259e2 | firme | cirrostratus, cirrocumulus |
| group_cb3f973a93f65557 | repetir | stratus, nimbostratus |
| group_cbdd3353378324aa | repetir | altocumulus, cirrocumulus |
| group_ced0317de0b525df | repetir | stratocumulus, cumulonimbus |
| group_d0114d09a6fd1d4e | firme | nimbostratus, stratocumulus |
| group_d07d01bd4dfce30f | firme | cirrus, cirrocumulus |
| group_d08e97f5ee3cc4ae | par | altocumulus, cirrostratus |
| group_d11febe9bf7778bb | repetir | nimbostratus, altocumulus |
| group_d208955cb7578b99 | par | cirrostratus, stratus |
| group_d3033f8a4a6bb2cc | repetir | altostratus, nimbostratus |
| group_d4e22606ccc5f24c | par | cirrostratus, altocumulus |
| group_d6a6e5d9f98cbc57 | repetir | stratocumulus, stratus |
| group_d9b30c7be46ce487 | repetir | cumulonimbus, cirrostratus |
| group_dbbcdebf6ecddd58 | repetir | altostratus, cumulonimbus |
| group_dd87c22b8c1e2334 | repetir | cirrocumulus, altostratus |
| group_de879b39ef05ce44 | par | altocumulus, cirrocumulus |
| group_e4296989052835fd | par | stratus, nimbostratus |
| group_e4e7327b81bcaff0 | firme | stratocumulus, nimbostratus |
| group_e56bc98d9e66865c | firme | nimbostratus, stratocumulus |
| group_e68dcda2529c190c | firme | stratocumulus, stratus |
| group_e90cbbde5454fcb1 | firme | cirrocumulus, altocumulus |
| group_ebf5f9b2aa5b3df1 | par | nimbostratus, altostratus |
| group_f0124bb78b773b8a | firme | altostratus, cirrus |
| group_f1d9c5357278a018 | par | cirrostratus, cirrus |
| group_f58466972ac5a207 | par | stratocumulus, stratus |
| group_f5d555a6e237c908 | firme | cirrostratus, cirrus |
| group_f5e71f7d888d53ee | par | altocumulus, nimbostratus |
| group_fa91da4f67dae921 | repetir | altostratus, no_nube |
| group_fb3cddd7aa4d4cb5 | par | cirrocumulus, stratus |
| group_fc6b8a67c2fb1cc9 | repetir | cirrostratus, stratocumulus |
| group_fffc6c4874436ebe | firme | cirrus, altocumulus |
