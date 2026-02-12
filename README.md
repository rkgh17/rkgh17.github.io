# Chirpy Starter

[![Gem Version](https://img.shields.io/gem/v/jekyll-theme-chirpy)][gem]&nbsp;
[![GitHub license](https://img.shields.io/github/license/cotes2020/chirpy-starter.svg?color=blue)][mit]

[RubyGems.org][gem]를 통해 [**Chirpy**][chirpy]를 설치하면, Jekyll은 테마 젬(gem)에 포함된 `_data`, `_layouts`, `_includes`, `_sass`, `assets` 폴더와 `_config.yml` 파일의 일부 옵션만 읽을 수 있습니다.
만약 이 테마 젬을 설치한 적이 있다면, `bundle info --path jekyll-theme-chirpy` 명령어를 사용하여 해당 파일들의 위치를 확인할 수 있습니다.

Jekyll 팀은 이것이 사용자의 자율성(ball in the user’s court)을 보장하기 위함이라고 설명하지만, 이는 결과적으로 사용자가 기능이 풍부한 테마를 사용할 때 '설치 즉시 완벽하게 작동하는(out-of-the-box)' 경험을 누리지 못하게 만듭니다.

Chirpy의 모든 기능을 온전히 사용하려면, 테마 젬에서 나머지 중요한 파일들을 직접 당신의 Jekyll 사이트로 복사해와야 합니다. 복사해야 할 대상 파일 목록은 다음과 같습니다.

```shell
.
├── _config.yml
├── _plugins
├── _tabs
└── index.html
```

여러분의 시간을 절약하고 복사 과정에서 파일이 누락되는 것을 방지하기 위해, 우리는 Chirpy 테마 최신 버전의 해당 파일/설정들과 CD(지속적 배포) 워크플로우를 이곳에 미리 추출해 두었습니다. 덕분에 여러분은 몇 분 만에 바로 글 작성을 시작할 수 있습니다.


## 사용법
[테마 문서](https://github.com/cotes2020/jekyll-theme-chirpy/wiki)를 확인해 주세요.

## 기여하기
이 저장소는 테마 저장소의 새 릴리스에 맞춰 자동으로 업데이트됩니다. 문제가 발생하거나 개선에 기여하고 싶다면, [테마 저장소][chirpy]를 방문하여 피드백을 남겨주세요.

## License

This work is published under [MIT][mit] License.

[gem]: https://rubygems.org/gems/jekyll-theme-chirpy
[chirpy]: https://github.com/cotes2020/jekyll-theme-chirpy/
[CD]: https://en.wikipedia.org/wiki/Continuous_deployment
[mit]: https://github.com/cotes2020/chirpy-starter/blob/master/LICENSE
