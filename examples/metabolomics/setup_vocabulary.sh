#!/bin/bash
# Convert metabolomics vocabulary to BMFM-compatible structure
# Required after running build_vocab.py

set -e

echo "Creating BMFM vocabulary structure..."

# Check prerequisite
if [ ! -f "bmfm_metabolomics_vocab/metabolites_vocab.txt" ]; then
    echo "ERROR: Run 'python build_vocab.py' first"
    exit 1
fi

# 1. Create main vocab.txt (special tokens + metabolites)
cat > bmfm_metabolomics_vocab/vocab.txt << 'VOCABEOF'
[UNK]
[SEP]
[PAD]
[CLS]
[MASK]
VOCABEOF
cat bmfm_metabolomics_vocab/metabolites_vocab.txt >> bmfm_metabolomics_vocab/vocab.txt

# 2. Create directory structure
mkdir -p all_genes_vocab/tokenizers/genes
mkdir -p all_genes_vocab/tokenizers/expressions

# 3. Copy main files
cp bmfm_metabolomics_vocab/vocab.txt all_genes_vocab/vocab.txt
cp bmfm_metabolomics_vocab/multifield_vocab.json all_genes_vocab/multifield_vocab.json
cp bmfm_metabolomics_vocab/vocab.txt all_genes_vocab/tokenizers/genes/vocab.txt

# 4. Create genes tokenizer config
VOCAB_SIZE=$(wc -l < all_genes_vocab/tokenizers/genes/vocab.txt)

cat > all_genes_vocab/tokenizers/genes/tokenizer_config.json << 'TOK'
{
  "do_lower_case": false,
  "unk_token": "[UNK]",
  "sep_token": "[SEP]",
  "pad_token": "[PAD]",
  "cls_token": "[CLS]",
  "mask_token": "[MASK]",
  "tokenizer_class": "BertTokenizer"
}
TOK

cat > all_genes_vocab/tokenizers/genes/config.json << CONF
{"vocab_size": $VOCAB_SIZE}
CONF

# 5. Create expressions tokenizer (bins 0-49)
cat > all_genes_vocab/tokenizers/expressions/vocab.txt << 'EXPR'
[UNK]
[SEP]
[PAD]
[CLS]
[MASK]
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
EXPR

cp all_genes_vocab/tokenizers/genes/tokenizer_config.json all_genes_vocab/tokenizers/expressions/
echo '{"vocab_size": 55}' > all_genes_vocab/tokenizers/expressions/config.json

echo "✓ Done! Created all_genes_vocab/ with $VOCAB_SIZE tokens"
