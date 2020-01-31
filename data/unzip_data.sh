#!/bin/bash
cat human_isoform.pos.gz* | zcat > human_isoform.pos
cat human_isoform.neg.gz* | zcat > human_isoform.neg
cat human_gene.pos.gz* | zcat > human_gene.pos
cat human_gene.neg.gz* | zcat > human_gene.neg
