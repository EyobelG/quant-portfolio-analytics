FROM golang:1.26-bookworm AS build

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -o /financial-advisor .

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /financial-advisor /financial-advisor

EXPOSE 8081
CMD ["/bin/sh", "-c", "/financial-advisor web --port ${PORT:-8081} api"]
