"""Builtin platform adapter registry."""

BUILTIN_ADAPTERS = {
    "telegram": "cliver.gateway.adapters.telegram.TelegramAdapter",
    "discord": "cliver.gateway.adapters.discord.DiscordAdapter",
    "slack": "cliver.gateway.adapters.slack.SlackAdapter",
    "feishu": "cliver.gateway.adapters.feishu.FeishuAdapter",
}
