# Trusted local editing

Edition means editing access. For a private single-user browser running on the
same computer, opt in with `viewer.trusted_local_editing: true` in the JSON
configuration (`"trusted_local_editing": true` inside `"viewer"`). The default
is false. This does not require or reveal an edition password.

Access is derived per request, never from a token claim: configuration must be
healthy, multi-user mode disabled and the browse password empty. The actual peer
must be loopback, with Host localhost, 127.0.0.1 or [::1]. Forwarded requests and
cross-origin/cross-site browser metadata are refused, including GET requests.
Existing password sessions and share-token scope keep their existing behavior.
Use direct local access; this option does not authorize network/reverse-proxy
access. An unmarked loopback proxy cannot be distinguished from a local
caller; do not enable this option behind a proxy. Browser Origin/Referer must
match the exact scheme, host and port.

The scan subprocess still requires its existing edition-password condition. The option neither edits ratings nor fills
passwords, changes tokens or enables multi-user administration.
