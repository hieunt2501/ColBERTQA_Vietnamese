CUDA_VISIBLE_DEVICES="1" \
python -m torch.distributed.launch --nproc_per_node=1 -m \
colbert.train --amp \
			--doc_maxlen 1024 \
			--mask-punctuation \
			--bsize 1 \
			--accum 1 \
			--query_maxlen 300 \
			--pretrained_tokenizer ./pretrained/bartpho \
            --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2 \
            --root ./root/pretrain