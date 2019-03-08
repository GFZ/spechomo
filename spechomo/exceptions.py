# -*- coding: utf-8 -*-


class ClassifierNotAvailableError(RuntimeError):
    def __init__(self, spechomo_method, src_sat, src_sen, src_LBA, tgt_sat, tgt_sen, tgt_LBA, n_clusters):
        self.spechomo_method = spechomo_method
        self.src_sat = src_sat
        self.src_sen = src_sen
        self.src_LBA = src_LBA
        self.tgt_sat = tgt_sat
        self.tgt_sen = tgt_sen
        self.tgt_LBA = tgt_LBA
        self.n_clusters = n_clusters
        RuntimeError.__init__(self)

    def __str__(self):
        return 'No %s classifier available for predicting %s %s %s from %s %s %s (%d clusters).'\
               % (self.spechomo_method, self.tgt_sat, self.tgt_sen, self.tgt_LBA,
                  self.src_sat, self.src_sen, self.src_LBA, self.n_clusters)
