# BD-LW: An Image Dataset and Multi-Criteria Benchmark for Local Weapon Recognition in Bangladesh

**Dataset (Mendeley Data):** [https://doi.org/10.17632/2ns343t8xg.1](https://doi.org/10.17632/2ns343t8xg.1)

BD-LW (Bangladeshi Local Weapon) is an image dataset of six traditional implements that are common in Bangladeshi households and farms and are also used as weapons in local violence.


## Dataset

The Mendeley archive contains **3,075** original photographs. After 125 near-duplicates are removed, **2,950** images form the benchmark.

| Label | Class | English | Original photos | Benchmark images |
|---|---|---|---|---|
| 0 | Churi | Knife | 530 | 521 |
| 1 | Dao | Machete | 533 | 530 |
| 2 | Khor | Straight razor | 536 | 471 |
| 3 | Kodal | Hoe | 482 | 477 |
| 4 | Kural | Axe | 486 | 477 |
| 5 | Shabol | Iron crowbar | 508 | 474 |
| | | **Total** | **3,075** | **2,950** |

| Subset | Content | Original photos | Benchmark images |
|---|---|---|---|
| Single | One weapon per image | 1,849 | 1,776 |
| Multi | Several weapons of the same class | 1,226 | 1,174 |

## Train, validation and test split

The benchmark uses one fixed split. Similar photographs (for example, several shots of the same object) are kept on the same side of the split, so no test image has a near-identical copy in the training data.

| Part | Images | Single | Multi |
|---|---|---|---|
| Test (locked) | 419 | 255 | 164 |
| Train / validation | 2,531 | 1,521 | 1,010 |
| **Total** | **2,950** | **1,776** | **1,174** |

The train/validation part is divided into five cross-validation folds. In each round, one fold is used for validation and the other four for training.

| Fold | Images | Single | Multi |
|---|---|---|---|
| 0 | 507 | 323 | 184 |
| 1 | 502 | 297 | 205 |
| 2 | 525 | 302 | 223 |
| 3 | 499 | 299 | 200 |
| 4 | 498 | 300 | 198 |

Test images per class: Churi 68, Dao 79, Khor 62, Kodal 67, Kural 63, Shabol 80.

## Split files

| File | Contents |
|---|---|
| `splits/manifest.csv` | The 2,950 benchmark images: image ID, original file (`source_file`), class, subset, capture source, split, fold (−1 for test) and similarity group |
| `splits/class_map.json` | Class names and labels |
| `splits/test.txt` | Image IDs of the test set |
| `splits/fold0.txt` … `fold4.txt` | Image IDs of each cross-validation fold |
| `splits/removed_duplicates.csv` | The 125 near-duplicates removed from the original photographs |

To use the split, match `source_file` in `manifest.csv` to the files in the Mendeley archive, train and validate on the images with `split == "trainval"` using the `fold` column, and report results on `split == "test"`.
