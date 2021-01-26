#!/usr/bin/env python3

import unittest

import cv2

import datasets


def dummy_mattr():
    class ModelAttribs():
        def __init__(self):
            self.inheight = 512
            self.inwidth = 512
            self.outheight = 256
            self.outwidth = 256
            self.nclasses = 2
    return ModelAttribs()


class TestDatasets(unittest.TestCase):
    def test_load_barcodes(self):
        ds_path = "/home/zastrow-marcks/mag/"\
            "barcodes/yolo/datasets/localization/"
        cv2.imwrite = lambda *args, **kwargs: True
        mattr = dummy_mattr()
        datasets.convert_annotations2img(ds_path, mattr)
