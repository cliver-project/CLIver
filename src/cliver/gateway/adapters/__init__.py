"""Builtin platform adapter registry."""

BUILTIN_ADAPTERS = {
    "telegram": "cliver.gateway.adapters.telegram.TelegramAdapter",
    "discord": "cliver.gateway.adapters.discord.DiscordAdapter",
    "slack": "cliver.gateway.adapters.slack.SlackAdapter",
    "wechat": "cliver.gateway.adapters.wechat.WeChatAdapter",
    "feishu": "cliver.gateway.adapters.feishu.FeishuAdapter",
}
