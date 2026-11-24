# Running Instinct in Docker

Running Instinct in a Docker container gives the agents a hard boundary. They
can only change the project folder you mount and the app's own data. Your
other files, home folder and system can't be seen or touched.

```bash
tools/docker/instinct.sh start --workspace ~/code/my-project
```

The first run builds the image. Instinct then opens in your browser, already
signed in. Press Ctrl-C to stop it; that also stops the models.

You need [Docker](https://docs.docker.com/get-docker/): Docker Desktop on a
Mac or Windows, or Docker Engine on Linux.

## Where the models run

**Mac.** Docker can't reach the Apple GPU, so Instinct is split in two:

- **On your Mac:** the models run natively at full speed. The launcher starts
  a model worker for Gemma 4 and one for Qwen3.5
  (`python -m instinct.container serve-models`). They listen on 127.0.0.1
  only, with a fresh token each run.
- **In the container:** the app, the agents, their commands, the terminal
  and your chats. It reaches the models through Docker's
  `host.docker.internal`.

You need the native setup first ([app.md](app.md): `.venv` and the models).
Quit the Instinct app before starting, because both load the same models and
they don't fit in memory together.

**Linux or Windows (WSL2) with an NVIDIA GPU.** Everything, models included,
runs in the container, with the GPU passed through (`--gpus all`). The models
are sized to the card the same way as [without Docker](nvidia.md#which-models-fit).
You need:

- the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (Docker Desktop on Windows includes GPU support for WSL2);
- the models downloaded once with `tools/nvidia/instinct.sh setup`. The
  container gets them read-only.

## What the container can do

| | |
| --- | --- |
| **Can change** | The workspace you mount (`/workspace`), the app's state folder (`/state`) and a 2 GB `/tmp` |
| **Can read** | The image (Python, git, the usual command-line tools) and the files it can change; nothing else from your computer |
| **Runs as** | An unprivileged user, with every Linux capability dropped and `no-new-privileges` |
| **Filesystem** | Read-only everywhere else; at most 1,024 processes |
| **Network** | The app's port is published on 127.0.0.1 only, so other machines can't reach it. Internet access stays on, so web search, `pip install` and `git clone` work inside the sandbox. |

**Where the state lives.** The state folder holds chats, settings and API
keys:

- on a Mac, `~/Library/Application Support/Instinct/docker`;
- on Linux, `~/.local/share/instinct-docker`.

**What's still exposed.** The agents can still change the project you mount,
so use git or a copy if that matters. They can also change the app's own
state. Inside the container, Instinct reports commands as sandboxed, and the
settings name the container as the sandbox.

**What isn't available.** Guest sharing is off inside Docker. Use the native
app to share.

## Options

```bash
tools/docker/instinct.sh start --workspace DIR --port 8765 --no-browser
tools/docker/instinct.sh build
```

- `--workspace` is the one folder the container can change. It defaults to
  `~/Instinct`.
- `--no-browser` prints the sign-in link instead of opening it. The link only
  works on this computer and only until Instinct stops.
- `build` rebuilds the image after you pull changes:
  - `instinct:app` (about 470 MB) on a Mac;
  - `instinct:nvidia` elsewhere.
- `INSTINCT_HOST_CONFIG=path.json` (Mac only) serves the models from a
  different worker configuration than `examples/instinct.json`.

**Building the images by hand:**

```bash
docker build -f docker/Dockerfile --target app -t instinct:app .
docker build -f docker/Dockerfile --target nvidia -t instinct:nvidia .
```

The launcher's `docker run` flags are the ones in the table above. To run the
image another way, keep them, publish the port on `127.0.0.1` only and set
`INSTINCT_APP_TOKEN` (32+ characters) for the sign-in.
