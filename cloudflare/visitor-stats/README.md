# Cloudflare Visitor Stats

这个 Worker 负责给 JoyRL Book 提供两类统计接口：

- `POST /track`：全站页面访问上报
- `GET /stats`：首页展示的 `今日访问 / 累计访问`

## 统计口径

- 当前统计的是全站 `PV`（页面浏览量），不是 `UV`
- 日期按 `Asia/Shanghai` 计算
- 数据存在 Cloudflare KV 里，重新部署 Worker 代码不会清空数据

## 为什么不会因为部署而重置

访问数据保存在独立的 `KV namespace` 里，而不是保存在 Worker 代码里。

只要后续部署继续绑定同一个 `KV namespace`，统计就会一直累积。真正会重置的情况只有：

- 你删除了原来的 KV namespace
- 你把绑定改到了新的 namespace
- 你手动清空了 KV 里的键

第一次用 CLI 部署时，Wrangler 会为 `VISITOR_STATS` 创建 KV，并把生成的 namespace ID 回写到 `wrangler.toml`。之后再次部署会复用同一个 namespace，所以不会因为发版而清零。

## 部署

1. 登录 Cloudflare

```bash
cd cloudflare/visitor-stats
npx wrangler login
```

2. 首次部署 Worker

```bash
npx wrangler deploy
```

部署成功后，你会得到一个类似下面的地址：

```text
https://joyrl-visitor-stats.<your-subdomain>.workers.dev
```

3. 把这个地址填到 GitHub 仓库变量 `VISITOR_STATS_API_URL`

- 路径：`Settings -> Secrets and variables -> Actions -> Variables`
- 变量名：`VISITOR_STATS_API_URL`
- 变量值：`https://joyrl-visitor-stats.<your-subdomain>.workers.dev`

4. 重新触发 GitHub Pages 部署

- 推一次代码到 `main`
- 或者手动重新运行 `Deploy Docusaurus to GitHub Pages`

## 本地调试

默认 `wrangler dev` 使用本地 KV，不会污染线上统计：

```bash
cd cloudflare/visitor-stats
npx wrangler dev
```

如果你确实想连线上 KV 调试，可以参考 Cloudflare 官方文档为 KV 绑定开启 remote 模式。

## 验证

先手动打一条访问：

```bash
curl -X POST "https://joyrl-visitor-stats.<your-subdomain>.workers.dev/track" \
  -H "Origin: https://datawhalechina.github.io" \
  -H "Content-Type: application/json" \
  --data '{"path":"/"}'
```

再查看聚合结果：

```bash
curl "https://joyrl-visitor-stats.<your-subdomain>.workers.dev/stats"
```

## 方案说明

Cloudflare 官方文档指出，KV 对“同一个 key”的写入限制是 `1 次/秒`，并且并发写入到同一 key 可能互相覆盖。为避开这个限制，这个 Worker 使用了分片计数器（sharded counters），把写入分散到多个 key 上，再在读取时汇总。

这套方案很适合文档站、社区站这类低到中等流量场景。如果以后流量明显变大，或者你想要更强一致的计数，可以升级到 Durable Objects 或 D1。
