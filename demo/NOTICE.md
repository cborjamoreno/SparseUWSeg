# Demo data — UCSD Mosaics

The six image/label pairs in this directory are **not ours**. They are tiles from the
**UCSD Mosaics** dataset, taken byte-for-byte from its training split:

| here | in the dataset |
|---|---|
| `original_1` | `FR3_512_1024_8192_8704` |
| `original_2` | `FR3_1024_1536_7680_8192` |
| `original_3` | `PALWave40_4096_4608_4608_5120` |
| `original_4` | `PALWave40_6656_7168_6144_6656` |
| `original_5` | `PALWave40_8192_8704_0_512` |
| `original_6` | `PALWave40_9216_9728_8704_9216` |

They are here only so `run.py` has something to smoke-test against. Cite the dataset if
you use it:

```bibtex
@article{edwards2017large,
  title={Large-area imaging reveals biologically driven non-random spatial patterns of
         corals at a remote reef},
  author={Edwards, Clinton B and Eynaud, Yoan and Williams, Gareth J and Pedersen,
          Nicole E and Zgliczynski, Brian J and Gleason, Arthur CR and Smith, Jennifer E
          and Sandin, Stuart A},
  journal={Coral Reefs}, volume={36}, number={4}, pages={1291--1305}, year={2017}
}

@article{alonso2019coralseg,
  title={CoralSeg: Learning coral segmentation from sparse annotations},
  author={Alonso, I{\~n}igo and Yuval, Matan and Eyal, Gal and Treibitz, Tali and
          Murillo, Ana C},
  journal={Journal of Field Robotics}, volume={36}, number={8}, pages={1456--1477},
  year={2019}
}
```

The dataset's own terms govern these files; check them before redistributing further.
