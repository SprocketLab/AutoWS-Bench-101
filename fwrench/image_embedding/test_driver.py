from fwrench.image_embedding.pca_embedding import PcaEmbedding

p = PcaEmbedding('tennis', 3)
ret = p.transform()
print("return valid set examples")
print(ret[1].examples)