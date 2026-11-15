import MarkdownIt from 'markdown-it'

/** 关闭原始 HTML，避免助手输出注入；外链统一新窗口并带 noopener。 */
const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  typographer: false,
})

const defaultLinkOpen = markdown.renderer.rules.link_open
markdown.renderer.rules.link_open = (tokens, index, options, env, self) => {
  const token = tokens[index]
  token?.attrSet('target', '_blank')
  token?.attrSet('rel', 'noopener noreferrer')
  return defaultLinkOpen
    ? defaultLinkOpen(tokens, index, options, env, self)
    : self.renderToken(tokens, index, options)
}

/** 渲染 Markdown；内容已在 markdown-it 层转义。 */
export function renderMarkdown(text: string): string {
  return markdown.render(text)
}
