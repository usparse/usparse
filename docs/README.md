# usparse documentation

usparse (ustb-sparse) is a sparse matrix library for GPUs (NVIDIA GPUs, AMD GPUs and Hygon DCUs).

## Installation

```bash
pnpm install
```

## Local Development

```bash
pnpm start
```
Or run with chinese language support:
```bash
pnpm start  --locale zh-Hans
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
pnpm build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true pnpm deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> pnpm deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
