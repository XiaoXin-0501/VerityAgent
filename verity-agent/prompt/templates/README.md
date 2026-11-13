# Prompt 模板目录

- `manifest.json`：prompt id → 文件/版本/变量声明；**模板正文不写进 manifest**。
- `rag/*.md`：当前已使用的 Prompt 正文。
- 业务代码只引用 `prompt/ids.py` 中的 `PromptId`，不直接写字符串 id。

## 新增 Prompt

1. 在 `manifest.json` 增加条目：`file / version / description / purpose / static / cacheable / variables`；
2. 新建 `.md` 正文，用 `${name}` 声明变量；
3. 在 `prompt/ids.py` 增加枚举值（`build_prompt_registry` 会校验枚举与 manifest 一致）；
4. 调用方通过 `PromptRegistry.render(PromptId.X, **variables)` 使用。

## 修改变量

- manifest 的 `variables` 必须与正文 `${name}` **完全一致**（多/少都在启动期报错）；
- 只有默认值的变量可设 `required=false` + `default`；领域画像未提供时使用默认值；
- 模板未声明的额外入参会被忽略，便于覆盖模板只使用变量子集。

## 静态 vs 动态

- 静态：`static=true`，`variables` 为空，无变量替换；
- 动态：声明变量，运行时渲染。

## 部署侧覆盖

外部目录结构与本目录一致（必须有 `manifest.json`），通过
`PROMPTS__DIRECTORY` 指定。覆盖项可只写 `file` 换正文，其余元数据继承内置；
完整规则见 `doc/Prompt模块设计.md`。
