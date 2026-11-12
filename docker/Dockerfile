FROM manimcommunity/manim:v0.18.1

# Non-root workspace for isolated ManimCE execution
USER root
RUN mkdir -p /workspace && chown -R 1000:1000 /workspace
USER 1000:1000

WORKDIR /workspace
ENTRYPOINT []
