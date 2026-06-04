# RuleDep Case Studies

本文档记录适合在 RuleDepDemo 中展示的 query-level 案例。这些案例都满足一个共同模式：Stage1 的 rank1 是一个有诱惑力但不够正确的候选 C1；Stage2 通过 rule dependency 把 GT 推到 rank1，同时把 C1 往后压。

## YAGO3-10: Laurie Calloway 效力 Southern California Lazers

这个例子比 FB15k-237 的 Israel/Eurasia case 更适合用来同时解释 complementarity 和 redundancy。GT 侧有一个很大的正 dependency：一条直接的 `isAffiliatedTo -> playsFor` 规则和一条“共同球队/职业轨迹”规则互补；C1 侧有一个很大的负 dependency：两条几乎相同的出生地代理规则互相冗余。

TikZ 图：

- `reports/query_analysis/figures/yago_laurie_calloway_tikz.tex`

Demo 文件：

- `RuleDepDemo/frontend/public/example/YAGO3-10/tg_r3d6__pos_auto_sqrt__ri_surprisal__dn_none__dl1_1e-5/playsFor/Laurie_Calloway_HEAD_.json`

Query：

```text
Laurie_Calloway playsFor ?
```

候选对比：

| candidate | 语义 | rank | Stage1 official | Stage2 official | dependency score | positive dep | negative dep |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Southern_California_Lazers | GT | 2 -> 1 | 0.136369 | 0.339155 | +1.351397 | 2 | 0 |
| Des_Moines_Menace | C1 | 1 -> 2 | 0.339225 | 0.185368 | -0.954824 | 3 | 2 |

这个 case 的语义比较好讲：`Laurie_Calloway` 在训练集中已经有 `isAffiliatedTo Southern_California_Lazers`，并且通过 `San_Jose_Earthquakes_(1974-88)` 与另一个也效力过 `Southern_California_Lazers` 的球员 `Charlie_Kadupski` 形成职业轨迹上的共同证据。C1 `Des_Moines_Menace` 也不是完全荒谬，因为训练集中有 `Laurie_Calloway isAffiliatedTo Des_Moines_Menace`；但 query 问的是 `playsFor`，而不是宽泛 affiliation。Stage2 的作用是：把 GT 侧的直接 affiliation 和轨迹证据合在一起加分，同时把 C1 侧由出生地 Birmingham 触发的重复代理证据压下去。

GT `Southern_California_Lazers` 的主要 rules：

```text
R39786,  weight=1.404430: playsFor(X,Y) <= isAffiliatedTo(X,Y)
R113088, weight=0.324349: playsFor(Laurie_Calloway,Y) <= isAffiliatedTo(Laurie_Calloway,Y)
R899565, weight=0.000249: playsFor(X,Y) <= playsFor(X,A), playsFor(B,A), playsFor(B,Y)
```

对应 grounding：

```text
R39786 / R113088:
  train: Laurie_Calloway isAffiliatedTo Southern_California_Lazers
  infer: Laurie_Calloway playsFor Southern_California_Lazers

R899565:
  train: Laurie_Calloway playsFor San_Jose_Earthquakes_(1974-88)
  train: Charlie_Kadupski playsFor San_Jose_Earthquakes_(1974-88)
  train: Charlie_Kadupski playsFor Southern_California_Lazers
  infer: Laurie_Calloway playsFor Southern_California_Lazers
```

GT 的 top displayed dependencies：

```text
R39786 + R899565:  +1.351342
R113088 + R899565: +0.000056
```

这里的关键是 `R899565` 本身权重几乎为 0，单独看只是很弱的“共同效力过某队”的轨迹规则；但它和 `isAffiliatedTo -> playsFor` 同时触发时，说明 direct affiliation 不是孤立噪声，而是与职业轨迹相互印证。因此 `R39786 + R899565` 得到很大的正 dependency，GT 的 `dependencyScore=+1.351397`，从 rank2 被推到 rank1。这个是比 Israel/Eurasia 更清楚的 complementarity。

C1 `Des_Moines_Menace` 的主要 rules：

```text
R39786,  weight=1.404430: playsFor(X,Y) <= isAffiliatedTo(X,Y)
R211342, weight=1.352040: playsFor(X,Des_Moines_Menace) <= isAffiliatedTo(X,Des_Moines_Menace)
R113088, weight=0.324349: playsFor(Laurie_Calloway,Y) <= isAffiliatedTo(Laurie_Calloway,Y)
R770881, weight=0.000000: playsFor(X,Y) <= wasBornIn(X,A), wasBornIn(B,A), isAffiliatedTo(B,Y)
R847023, weight=0.000000: playsFor(X,Y) <= wasBornIn(X,A), wasBornIn(B,A), playsFor(B,Y)
```

对应 grounding：

```text
R39786 / R211342 / R113088:
  train: Laurie_Calloway isAffiliatedTo Des_Moines_Menace
  infer: Laurie_Calloway playsFor Des_Moines_Menace

R770881:
  train: Laurie_Calloway wasBornIn Birmingham
  train: Mickey_Lewis wasBornIn Birmingham
  train: Mickey_Lewis isAffiliatedTo Des_Moines_Menace
  infer: Laurie_Calloway playsFor Des_Moines_Menace

R847023:
  train: Laurie_Calloway wasBornIn Birmingham
  train: Mickey_Lewis wasBornIn Birmingham
  train: Mickey_Lewis playsFor Des_Moines_Menace
  infer: Laurie_Calloway playsFor Des_Moines_Menace
```

C1 的 displayed dependencies：

```text
R770881 + R847023: -1.320266
R39786  + R770881: -0.380889
R39786  + R847023: +0.745964
R113088 + R770881: +0.000184
R113088 + R847023: +0.000184
```

这里最重要的是 `R770881 + R847023=-1.320266`。这两条规则几乎只差最后一步用 `isAffiliatedTo(B,Y)` 还是 `playsFor(B,Y)`，但前半段完全相同：`Laurie_Calloway` 和 `Mickey_Lewis` 都出生在 Birmingham。它们不是两条独立证据，而是同一个出生地代理模式的重复表达。Stage2 因此学到强负 dependency，把 C1 的 `dependencyScore` 压到 `-0.954824`，使 `Des_Moines_Menace` 从 rank1 掉到 rank2。

这个例子适合在图中讲成一句话：

```text
GT: direct affiliation + shared career path => complementarity
C1: two birthplace-proxy paths => redundancy
```

## FB15k-237: Israel 被 Eurasia 包含

Demo 文件：

- `RuleDepDemo/frontend/public/example/FB15k-237/tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5/_location_location_contains/_TAIL__m_03spz.json`

Query：

```text
? /location/location/contains Israel
```

候选对比：

| candidate | 语义 | rank | Stage1 official | Stage2 official | dependency score | positive dep | negative dep |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eurasia | GT | 2 -> 1 | 0.646172 | 0.646461 | +0.393945 | 1939 | 785 |
| Africa | C1 | 1 -> 2 | 0.706941 | 0.618577 | -1.194277 | 9 | 232 |

作图和解释 rank 时应使用 `Stage1 official` / `Stage2 official`，不要直接使用 demo JSON 中的 raw `stage1` / `stage2` logit。这个 case 中 Eurasia 的 raw rule logit 很高，但它的 `maxConf=0.647059`，official score 被 cap 在 0.646 附近；Africa 的 `maxConf=0.753731` 更高，所以 Stage1 official 仍然是 Africa 排在 Eurasia 前面。Stage2 的主要翻转来自 Africa 被负 dependency 从 0.706941 压到 0.618577。

这个例子的语义点是：Eurasia 不是中东，但 Middle East / Israel 位于西亚，西亚属于 Eurasia，所以 `Eurasia contains Israel` 合理；`Africa contains Israel` 不合理。需要注意，Africa 的错误不是一个“所有 Egypt 邻国都会被分到 Africa”的普遍规律，而是这个 query 中若干 `contains/adjoins` 规则共同触发后，把“邻接”错当成了“包含”。

GT `Eurasia` 的 top-weight displayed rules：

```text
R1344261, weight=1.190834: contains(X,Y) <= contains(X,A), adjoins(A,Y)
R451157,  weight=1.142168: contains(X,Y) <= contains(X,A), adjoins(Y,A)
R1724139, weight=0.710134: contains(Eurasia,Y) <= contains(Asia,Y)
```

其中 `contains(Asia, Israel)` 在训练集中存在，所以 `contains(Asia,Y) -> contains(Eurasia,Y)` 是最容易解释的主证据。前两条是通用的“如果 X 包含 A，A 与 Y 相邻，则 X 可能包含 Y”规则；它们在地理区域 query 中权重很高，但也可能造成误导。

在这个具体 query 上，GT 的 top rules 有明确 grounding：

```text
R1724139:
  train: Asia contains Israel
  infer: Eurasia contains Israel

R1344261:
  train: Eurasia contains Syria,  Syria adjoins Israel
  train: Eurasia contains Jordan, Jordan adjoins Israel
  train: Eurasia contains Lebanon, Lebanon adjoins Israel

R451157:
  train: Eurasia contains Syria, Israel adjoins Syria
```

所以 GT 的证据链不是“Eurasia 是中东”，而是“Israel 位于 Asia；并且 Syria/Jordan/Lebanon 这些 Eurasia 内部地点与 Israel 相邻”。这组证据在语义上可以解释为：邻接规则在 Eurasia 这种大区域上是合理泛化，因为被邻接的 A 本来就在 Eurasia 内。

GT `Eurasia` 的 top displayed dependencies：

```text
R259938 + R1724139: +0.001912
  R259938:  contains(Eurasia,Y) <= film_release_region(The_Tourist,Y)
  R1724139: contains(Eurasia,Y) <= contains(Asia,Y)

R684375 + R1557767: +0.001585
  R684375:  contains(Eurasia,Y) <= olympic_athlete_affiliation_country(Archery,Y)
  R1557767: contains(Eurasia,Y) <= film_release_region(Eternal_Sunshine_of_the_Spotless_Mind,Y)

R684375 + R1642231: +0.000921
  R684375:  contains(Eurasia,Y) <= olympic_athlete_affiliation_country(Archery,Y)
  R1642231: contains(Eurasia,Y) <= film_release_region(The_Expendables,Y)

R1642231 + R1719657: -0.007022
  R1642231: contains(Eurasia,Y) <= film_release_region(The_Expendables,Y)
  R1719657: contains(Eurasia,Y) <= olympic_athlete_affiliation_country(Rhythmic_gymnastics,Y)
```

Eurasia 的 displayed dependency 单项都不大，真正重要的是数量：候选一共有 200 条 scored rules、1939 个正 dependency 和 785 个负 dependency，合计 `dependencyScore=+0.393945`。因此它不是靠一条特别大的 dependency 翻盘，而是靠大量小的正向依赖累积。图中可以把 GT 侧概括为：主要规则相对独立，少数 dependency 起到互补式加分。这里还要区分 rule weight 和 dependency weight：`R1724139` 本身是强规则，但它和其他 displayed rule 的 dependency 较小；Stage2 对 GT 的正向作用主要体现在许多小依赖累加，而不是某一个 dependency pair 特别大。

C1 `Africa` 的 top-weight displayed rules：

```text
R1344261, weight=1.190834: contains(X,Y) <= contains(X,A), adjoins(A,Y)
R451157,  weight=1.142168: contains(X,Y) <= contains(X,A), adjoins(Y,A)
R1030243, weight=0.525395: contains(X,Y) <= contains(X,A), adjoins(B,A), adjoins(B,Y)
R397615,  weight=0.055459: contains(X,Y) <= countries_within(X,A), adjoins(Y,A)
R1291877, weight=0.045325: contains(X,Y) <= countries_within(X,A), adjoins(A,Y)
```

Africa 的 top negative displayed dependencies：

```text
R661328 + R950801: -0.015602
  R661328: contains(X,Y) <= countries_within(X,A), adjoins(A,B), adjoins(B,Y)
  R950801: contains(X,Y) <= countries_within(X,A), adjoins(B,A), adjoins(B,Y)

R210865 + R950801: -0.014282
  R210865: contains(X,Y) <= countries_within(X,A), adjoins(B,A), adjoins(Y,B)
  R950801: contains(X,Y) <= countries_within(X,A), adjoins(B,A), adjoins(B,Y)

R210865 + R661328: -0.013849
  R210865: contains(X,Y) <= countries_within(X,A), adjoins(B,A), adjoins(Y,B)
  R661328: contains(X,Y) <= countries_within(X,A), adjoins(A,B), adjoins(B,Y)
```

它们的直觉是：如果 continent X 里有国家 A，A 或与 A 相邻的 B 又与 Y 相邻，那么 X 可能包含 Y。对 Africa/Israel，训练图中实际触发了 Egypt 相关路径，例如 Egypt 与 Israel 相邻，Libya/Sudan 与 Egypt 相邻。这类路径本质上只说明“邻接”，不是“包含”。Stage2 识别出这些邻接推理之间高度重叠且误导，给 Africa 累积了 232 个负 dependency，最终把它从 rank1 压到 rank2。

C1 的具体 grounding 更能说明为什么 Stage1 会先选错 Africa：

```text
R1344261:
  train: Africa contains Egypt, Egypt adjoins Israel

R451157:
  train: Africa contains Egypt, Israel adjoins Egypt

R1291877:
  train: Africa countries_within Egypt, Egypt adjoins Israel

R397615:
  train: Africa countries_within Egypt, Israel adjoins Egypt

R950801 / R661328 / R210865:
  train: Africa countries_within Sudan, Egypt adjoins Sudan, Egypt adjoins Israel
  train: Africa countries_within Libya,  Egypt adjoins Libya,  Egypt adjoins Israel
```

这解释了两个现象：

1. Stage1 为什么把 Africa 排到 rank1：`Africa contains Egypt` 加上 `Egypt adjoins Israel` 会触发和 GT 类似的高权重邻接规则，尤其 `R1344261/R451157`。
2. Stage2 为什么把 Africa 压下去：Africa 的后几条规则都是同一个错误模式的变体，即通过 `countries_within(Africa,A)` 和一跳/两跳 `adjoins` 把 Israel 扩散到 Africa。`R661328/R950801/R210865` 共享同样的 Sudan/Libya/Egypt/Israel grounding，所以 dependency 学到它们是冗余且不可靠的组合，三个 displayed pair 全是负数。

为了验证这个错误是否会泛化到其他国家，我检查了 Greece、Jordan、Saudi Arabia。它们没有对应的 RuleDepDemo query 文件，所以不能直接给 demo rank；但按上面 Africa-specific 规则在训练图中查触发情况如下：

| target country | direct Africa rule trigger | two-hop Africa rule trigger | note |
| --- | ---: | ---: | --- |
| Greece | 0 | 0 | 不会由这些 Egypt/Africa 邻接规则触发。 |
| Jordan | 0 | 1 | 有一条 `Jordan -> Israel -> Egypt` 的两跳邻接路径，但比 Israel 弱很多。 |
| Saudi Arabia | 0 | 0 | 不会由这些 Africa 邻接规则触发。 |

所以更准确的说法是：这个 case 里 Africa 的错误主要来自 Israel 与 Egypt 的直接陆地邻接，以及 Egypt 与其他非洲国家的邻接链条；不是所有中东或地中海国家都会被同样推成 Africa。FB15k-237 的 `adjoins` 关系也有明显的不完整/不对称现象，例如训练集中有 `Jordan -> Israel`，但没有对应的 `Israel -> Jordan`。

需要注意的是，Eurasia 的 Stage2 分数只从 0.646172 增到 0.646461，变化很小，因为它接近 `maxConf=0.647059` 的上限；本例排名翻转主要来自 Africa 被明显压低。

## YAGO3-10: Battle of Changsha 发生在 Hunan

这个 case 比 football affiliation 更适合作为主 YAGO 示例，因为语义更直观：`Battle_of_Changsha_(1944)` 是发生在湖南/长沙一带的具体战役；`Chinese_Civil_War` 是更大范围的战争，训练集中主要连到 `China`、`Taiwan`、`Jiangxi`，不是 `Hunan`。

Demo 文件：

- `RuleDepDemo/frontend/public/example/YAGO3-10/tg_r3d6__pos_auto_sqrt__ri_surprisal__dn_none__dl1_1e-5/happenedIn/_TAIL_Hunan.json`

Query：

```text
? happenedIn Hunan
```

候选对比：

| candidate | 语义 | rank | Stage1 official | Stage2 official | dependency score | positive dep | negative dep |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Battle_of_Changsha_(1944) | GT | 2 -> 1 | 0.097898 | 0.113284 | +0.322807 | 21 | 15 |
| Chinese_Civil_War | C1 | 1 -> 2 | 0.115430 | 0.072124 | -0.605140 | 11 | 11 |

原始数据中有直接证据：

```text
test:  Battle_of_Changsha_(1944) happenedIn Hunan
train: Battle_of_Changsha_(1944) isLocatedIn Hunan
train: Battle_of_Changsha_(1944) happenedIn Changsha
train: Battle_of_Changsha_(1944) happenedIn Hengyang
train: Hunan hasCapital Changsha
```

C1 的背景也能解释 Stage1 为什么会被吸引：

```text
train: Chinese_Civil_War happenedIn China
train: Chinese_Civil_War happenedIn Taiwan
train: Chinese_Civil_War happenedIn Jiangxi
```

也就是说，Stage1 先选择了一个“大范围中国历史事件”；Stage2 则更偏向具体发生在 Hunan/Changsha/Hengyang 的战役。

GT `Battle_of_Changsha_(1944)` 的 top-weight displayed rules：

```text
R585262, weight=0.818521: happenedIn(X,Y) <= happenedIn(X,A), hasCapital(Y,A)
R481318, weight=0.597328: happenedIn(X,Y) <= happenedIn(X,A), isLocatedIn(A,Y)
R17244,  weight=0.000490: happenedIn(X,Y) <= participatedIn(A,X), participatedIn(A,B), happenedIn(B,Y)
```

前两条语义很清楚：

- 如果事件 X 发生在 A，而 A 是 Y 的首府，则 X 可能发生在 Y。这里可对应 `Battle_of_Changsha_(1944) happenedIn Changsha` 与 `Hunan hasCapital Changsha`。
- 如果事件 X 发生在 A，而 A 位于 Y，则 X 可能发生在 Y。这里可对应 `Changsha/Hengyang isLocatedIn Hunan`。

GT 的 top positive dependencies 及对应规则：

```text
R17244 + R433628: +0.258741, synergy
  R17244:  happenedIn(X,Y) <= participatedIn(A,X), participatedIn(A,B), happenedIn(B,Y)
  R433628: happenedIn(X,Y) <= happenedIn(X,A), isLocatedIn(B,A), happenedIn(B,Y)

R67778 + R481318: +0.220157, synergy
  R67778:  happenedIn(X,Y) <= happenedIn(X,A), happenedIn(B,A), happenedIn(B,Y)
  R481318: happenedIn(X,Y) <= happenedIn(X,A), isLocatedIn(A,Y)

R17244 + R481318: +0.179259, synergy
  R17244:  happenedIn(X,Y) <= participatedIn(A,X), participatedIn(A,B), happenedIn(B,Y)
  R481318: happenedIn(X,Y) <= happenedIn(X,A), isLocatedIn(A,Y)
```

这些 dependency 的共同点是：`happenedIn`、`isLocatedIn`、`participatedIn` 三类证据互相配合。它们不是只凭一个地名相似性，而是把“战役发生地”和“地点层级”组合起来，因此把 GT 往前推。

C1 `Chinese_Civil_War` 的 top-weight displayed rules：

```text
R759663, weight=1.298334: happenedIn(X,Y) <= participatedIn(A,X), isAffiliatedTo(B,A), isPoliticianOf(B,Y)
R341403, weight=0.238736: happenedIn(X,Y) <= happenedIn(X,A), isLocatedIn(B,A), hasCapital(Y,B)
R17244,  weight=0.000490: happenedIn(X,Y) <= participatedIn(A,X), participatedIn(A,B), happenedIn(B,Y)
```

C1 的 strongest negative dependencies：

```text
R17244 + R678889: -0.238733, synergy
  R17244:  happenedIn(X,Y) <= participatedIn(A,X), participatedIn(A,B), happenedIn(B,Y)
  R678889: happenedIn(X,Y) <= happenedIn(X,A), participatedIn(A,B), happenedIn(B,Y)

R17244 + R82544: -0.229988, synergy
  R17244: happenedIn(X,Y) <= participatedIn(A,X), participatedIn(A,B), happenedIn(B,Y)
  R82544: happenedIn(X,Y) <= participatedIn(A,X), happenedIn(B,A), happenedIn(B,Y)

R67778 + R536952: -0.222473, synergy
  R67778:  happenedIn(X,Y) <= happenedIn(X,A), happenedIn(B,A), happenedIn(B,Y)
  R536952: happenedIn(X,Y) <= happenedIn(X,A), happenedIn(B,A), isLocatedIn(B,Y)
```

这些负 dependency 主要压的是“通过参战者/其他事件把大范围战争传播到局部地点”的路径。对 `Chinese_Civil_War` 来说，训练集中有 `happenedIn China/Taiwan/Jiangxi`，但没有强的 Hunan 直接证据；这类广域战争规则之间互相重叠后，Stage2 把它从 0.115430 压到 0.072124。

## YAGO3-10: Todsaporn Sri-reung 的俱乐部归属（备选）

这个 case 数值上满足条件，但语义上不如 `happenedIn/Hunan` 清楚：球员与国家队的 `isAffiliatedTo` 可能存在标注灰区。因此它更适合作为 football relation 的备选机制例子，而不是主展示例子。

Demo 文件：

- `RuleDepDemo/frontend/public/example/YAGO3-10/tg_r3d6__pos_auto_sqrt__ri_surprisal__dn_none__dl1_1e-5/isAffiliatedTo/Todsaporn_Sri-reung_HEAD_.json`

Query：

```text
Todsaporn_Sri-reung isAffiliatedTo ?
```

候选对比：

| candidate | 语义 | rank | Stage1 official | Stage2 official | dependency score | positive dep | negative dep |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Singhtarua_F.C. | GT | 2 -> 1 | 0.270689 | 0.378758 | +0.538756 | 27 | 13 |
| Thailand_national_football_team | C1 | 1 -> 2 | 0.368993 | 0.112326 | -1.811155 | 36 | 29 |

原始数据中有直接证据：

```text
train: Todsaporn_Sri-reung playsFor Singhtarua_F.C.
test:  Todsaporn_Sri-reung isAffiliatedTo Singhtarua_F.C.
```

同时，训练集中也有容易造成误导的背景：

```text
Todsaporn_Sri-reung playsFor Thailand_national_under-23_football_team
Todsaporn_Sri-reung isAffiliatedTo Thailand_national_under-23_football_team
Todsaporn_Sri-reung wasBornIn Bangkok
```

所以 Stage1 把 `Thailand_national_football_team` 放到 rank1 是可以理解的：它把 U23 国家队、出生地 Bangkok、以及泰国球员常见的国家队关联泛化到了 senior national team。

支持 GT `Singhtarua_F.C.` 的主要规则：

```text
R185849: isAffiliatedTo(X,Singhtarua_F.C.) <= playsFor(X,Singhtarua_F.C.)
R928372: isAffiliatedTo(X,Y) <= playsFor(X,Y)
```

这两条规则都直接表达了“如果球员为某队效力，那么他 affiliated to 该队”。此外，一组共现/队友型规则提供正 dependency，例如：

```text
R183019 + R928372: +1.366513, synergy
  R183019: isAffiliatedTo(X,Y) <= isAffiliatedTo(X,A), isAffiliatedTo(B,A), isAffiliatedTo(B,Y)
  R928372: isAffiliatedTo(X,Y) <= playsFor(X,Y)

R501919 + R928372: +1.292714, synergy
  R501919: isAffiliatedTo(X,Y) <= isAffiliatedTo(X,A), playsFor(B,A), isAffiliatedTo(B,Y)
  R928372: isAffiliatedTo(X,Y) <= playsFor(X,Y)

R266368 + R928372: +0.850496, synergy
  R266368: isAffiliatedTo(X,Y) <= isAffiliatedTo(X,A), playsFor(B,A), playsFor(B,Y)
  R928372: isAffiliatedTo(X,Y) <= playsFor(X,Y)
```

这些规则的形式大致是：候选人已有某些队伍 affiliation，其他人与这些队伍或目标队伍之间存在 `playsFor/isAffiliatedTo` 共现，于是目标队伍得到额外支持。单条规则未必漂亮，但和直接的 `playsFor -> isAffiliatedTo` 规则一起出现时，dependency 把它们识别为对 GT 有用的组合信号。

压低 C1 `Thailand_national_football_team` 的主要规则和 dependency：

```text
R597670: isAffiliatedTo(X,Thailand_national_football_team) <= isAffiliatedTo(X,Thailand_national_under-23_football_team)
R908980: isAffiliatedTo(X,Thailand_national_football_team) <= playsFor(X,Thailand_national_under-23_football_team)
R745264: isAffiliatedTo(X,Thailand_national_football_team) <= wasBornIn(X,Bangkok)

R160679 + R776835: -1.286940, synergy
  R160679: isAffiliatedTo(X,Y) <= wasBornIn(X,A), wasBornIn(B,A), playsFor(B,Y)
  R776835: isAffiliatedTo(X,Y) <= wasBornIn(X,A), wasBornIn(B,A), isAffiliatedTo(B,Y)

R776835 + R875357: -0.000114, synergy
  R776835: isAffiliatedTo(X,Y) <= wasBornIn(X,A), wasBornIn(B,A), isAffiliatedTo(B,Y)
  R875357: isAffiliatedTo(X,Thailand_national_football_team) <= playsFor(X,BEC_Tero_Sasana_F.C.)

R875357 + R908980: -0.0000369, redundancy
  R875357: isAffiliatedTo(X,Thailand_national_football_team) <= playsFor(X,BEC_Tero_Sasana_F.C.)
  R908980: isAffiliatedTo(X,Thailand_national_football_team) <= playsFor(X,Thailand_national_under-23_football_team)
```

C1 的规则看起来很有诱惑力：U23 国家队、Bangkok 出生地、以及曾效力于泰国俱乐部都可能暗示 senior national team。但这些证据太宽，且互相重叠。特别是 `wasBornIn` 相关的两跳规则：

```text
R160679: isAffiliatedTo(X,Y) <= wasBornIn(X,A), wasBornIn(B,A), playsFor(B,Y)
R776835: isAffiliatedTo(X,Y) <= wasBornIn(X,A), wasBornIn(B,A), isAffiliatedTo(B,Y)
```

这两条相当于“同出生地的人 B 曾效力/隶属于 Y，所以 X 也可能隶属于 Y”。对 `Thailand_national_football_team` 这种泛化目标来说，这种信号非常容易过度扩散。Stage2 因此给 C1 累积了 29 个负 dependency，把它从 0.368993 压到 0.112326。

这个 YAGO 例子的展示重点可以放在：Stage1 被“U23 国家队/出生地/国家队泛化”吸引，Stage2 则更相信直接的 `playsFor Singhtarua_F.C. -> affiliatedTo Singhtarua_F.C.`，并把过宽的国家队泛化规则降权。
