from fwrench.image_embedding.pca_embedding import fit

ret = fit(dataset='tennis', num_c=3)
print("return valid set examples")
print(ret[1].examples)