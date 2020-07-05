# LearnBayesNN
ベイズ深層学習の自学コード

[TensorFlow Probability](https://www.tensorflow.org/probability) (以下tfp)を利用しての、深層ベイズ学習の実装方法を整理する。

以下のように `import` しておく。
```python
import tensorflow_probability as tfp
```

## 1. 確率分布
tfp では、 [`tfp.distributions`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions) の中に、様々な基本的な確率分布を表現するクラスが実装されている。

主なメソッド
- `sample(sample_shape=(), seed=None, name='sample', **kwargs)` 分布に従う乱数を取得
- `prob(value, name='prob', **kwargs)` 尤度
- `log_prob(value, name='log_prob', **kwargs)` 対数尤度

### 1.1 同時確率
tfpでは、[`tfp.distributions.JointDistribution`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JointDistribution) を継承した3つのクラスを利用する実装方法がある。

* [`JointDistributionSequence`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JointDistributionSequential)
  * keras の `Sequence` のように`list`で実装する
  * 注意点は、逆順に引数に渡されること。
  * `sample()` メソッドの戻り値も、 `list`
* [`JointDistributionNamed`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JointDistributionNamed)
  * `dict` として、変数と分布をペアにして設定する
  * `sample()` メソッドの戻り値も `dict` 形式となる
* [`JointDistributionCoroutine`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JointDistributionCoroutine)
  * コルーチンの `yield` を利用して実装する
  * `sample()` の戻り値は、 `tuple`

## 2. 変分推論

## 3. MCMC

## 4. 深層学習での利用

[`tfp.layers`](https://www.tensorflow.org/probability/api_docs/python/tfp/layers) に実装されているレイヤーが `tf.keras` で利用することができるベイズ用のレイヤー

### 4.1 Flipout
Flipout をクラス名に含んでいるクラス (例: [`DenseFlipout`](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout)) は、ミニバッチ内の各サンプルに異なる重みを利用する手法。[2]

再パラメータ化より、およそ2倍計算量があるが、varianceが小さくなる。

# 参照
* [1] D. Piponi _et al_., "Joint Distributions for TensorFlow Probability",arXiv cs.PL 2001.11819 (2020) https://arxiv.org/abs/2001.11819
* [2] Y. Wen _et al_.,  "Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches", arXiv cs.LG 1803.04386 (2018) https://arxiv.org/abs/1803.04386
