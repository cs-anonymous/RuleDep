# Case Study Examples

Examples where the stage-1 model failed (`rank > 1`) but the dependency-augmented final model succeeded (`rank = 1`).

| dataset | relation | zh | direction | query | gold | stage1 top1 | final top1 | stage1 rank | final rank | rank gain |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| hetionet | DrD | 疾病与疾病相似 | tail | (Disease::DOID:11615, DrD, ?) | Disease::DOID:11054 | Disease::DOID:4045 | Disease::DOID:11054 | 5.0 | 1.0 | 4.0 |
| hetionet | DrD | 疾病与疾病相似 | head | (?, DrD, Disease::DOID:1245) | Disease::DOID:1964 | Disease::DOID:11239 | Disease::DOID:1964 | 3.0 | 1.0 | 2.0 |
| hetionet | DrD | 疾病与疾病相似 | tail | (Disease::DOID:2174, DrD, ?) | Disease::DOID:4159 | Disease::DOID:119 | Disease::DOID:4159 | 3.0 | 1.0 | 2.0 |
