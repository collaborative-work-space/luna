# Third-Party Notices

This project is built on top of, and vendors code from, the following projects.

## Panoptic Lifting (base framework)

`src/dataset`, `src/model`, `src/trainer`, `src/inference`, and `src/util` extend the
Panoptic Lifting codebase (Copyright (c) Meta Platforms, Inc., as marked in file headers):

> Y. Siddiqui, L. Porzi, S. R. Bulò, N. Müller, M. Nießner, A. Dai, and P. Kontschieder,
> "Panoptic Lifting for 3D Scene Understanding with Neural Fields," CVPR 2023.

Please consult the original [Panoptic Lifting repository](https://github.com/nihalsid/panoptic-lifting)
for its license terms before redistributing this derivative code.

## FAMO (vendored, unmodified)

`src/FAMO/famo.py` is copied unmodified from https://github.com/Cranial-XIX/FAMO under the
MIT License:

```
MIT License

Copyright (c) 2023 Bo Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Citation:
> B. Liu, Y. Feng, P. Stone, and Q. Liu, "FAMO: Fast Adaptive Multitask Optimization,"
> NeurIPS 2023, vol. 36, pp. 57226–57243.
