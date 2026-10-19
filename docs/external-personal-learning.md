# External personal-learning policy

The default `auto_retrain.backend` is `"native"`, preserving existing local
personal-ranker and keeper-head behavior. Set `"backend": "external"` inside
`"auto_retrain"` to retain preferences and pending comparison counts without
starting local personal-learning or evaluating learned heads on the CPU.

This is a dispatch policy, not an external job service. It does not launch an
external GPU worker or claim that a personal model has been trained. A separate
operator-controlled training workflow must authenticate observations, qualify
its model and publish the resulting scores. Existing ratings and comparisons
remain available for that workflow.

In external mode, notifications increment the existing durable counters without
consuming them or scheduling a training thread. Queued workers recheck policy
and return their consumed pending count when deferred. Manual personal-ranker,
keeper-head and weight-learning entry points return an explicit deferred result.
Keeper probabilities are unavailable; Compare learning/confidence responses
report `available: false` and an explanatory message.

A malformed or unreadable backend policy fails closed to external mode. A valid
configuration with no backend key uses native mode. The viewer reads its already
resolved server configuration path; CLI calls honor their explicit configuration.
This option does not disable ordinary deterministic photo scoring or erase votes.
