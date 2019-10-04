Predicting spectral information / multi-sensor homogenization
-------------------------------------------------------------




im_homo = predict_image(src_sat, src_sen, tgt_sat, tgt_sen, im_basename,
                                                method=method, n_clusters=n_clust, classif_alg=classif_alg)

                    im_homo.save(outpath)
