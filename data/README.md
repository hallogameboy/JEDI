# Experimental Datasets for Circular RNA Prediction

## General Instructions

Since there is a 100M maximum file size on GitHub, we compressed the experimental datasets into small zipped files. Please enter this directory and execute the script ./unzip_data.sh to merge and unzip the files.

```
cd data/
./unzip_data.sh
```

## Dataset Description

### Naming

Each file is named as **[dataset]**.**[class]** over three datasets and two classes.

**[dataset]** is one of the datasets:
1. **human_gene**
2. **human_isoform**:
3. **mouse**

**[class]** 
1. **pos**:
2. **neg**

### Data Format

All of the files are in the following JSON format.

Each line represents a gene as a training/testing instance with its file naming as the class.
```
{
  "junctions": {
    "head": [0, 2261,2923, ...],
    "tail": [283, 2657, 3028, ...]
  },
  "strand": "-",
  "seq": "TTTCAGACTC...",
  ...
}
```

Although there are more information, such as gene IDs and chromosone numbers in the datasets, JEDI only requires the fields of **"junctions"**, **"strand"**, and **"seq"** to construct the input data.

* **"strand"** maps to the strand of the gene as a string.
* **"junctions"** maps a dict with the keys **"head"** and **"tail"** mapping to two same-length integer lists.
  * The list of **"head"** represents the starting offsets of splice junctions.
  * The list of **"tail"** represents the end offsets of splice junctions.
* **"seq"** maps to the corresponding nucleotide sequence.

Note that the offets are based on the positive strand, so the sequence should be reversed before using the offets to query flanking regions if the sequence is on the negative strand.

## Experimental Setup

In our paper, we have two main tasks and one independent study using the following experimental setup with datasets for 5-fold cross-validation.

### Isoform-level Circular RNA Prediction

For each fold:

| Training | Testing |
| ----------- | ----------- |
| 80% of human_isoform.* | 20% of human_isoform.*  |


### Gene-level Circular RNA Prediction

For each fold:

| Training | Testing |
| ----------- | ----------- |
| 80% of human_gene.* | 20% of human_gene.*  |

### Independent Study

For each fold:

| Training | Testing |
| ----------- | ----------- |
| 100% of human_isoform.* | 100% of mouse.*  |



## Data File Checksum

Here we provide the checksums of all files for verification.

```
$ md5sum *
a797caf1bd75d33f48d337fa1de4a56f  human_gene.neg
c4b6027943bd704c18f6c4b0dcab858a  human_gene.pos
1e4e9ee9fb5b31c8a8f7b4a2ee5e550e  human_isoform.neg
6da5b581b9a6ad221f67685538b83727  human_isoform.pos
b247d0074589a8084d5654912fba99bf  mouse.neg
b23c26d0b62d5d310cb1baa68515d89a  mouse.pos
```
