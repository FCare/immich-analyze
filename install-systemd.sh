#!/bin/bash
# install-systemd.sh - Installation automatique des services systemd pour immich-analyze

set -e

PROJECT_DIR="$(pwd)"
USER_NAME="${SUDO_USER:-$(whoami)}"

echo "🚀 Installation des services systemd pour immich-analyze..."
echo "📁 Répertoire du projet: $PROJECT_DIR"
echo "👤 Utilisateur: $USER_NAME"

# Vérification des permissions
if [[ $EUID -ne 0 ]]; then
   echo "❌ Ce script doit être exécuté avec sudo"
   echo "💡 Usage: sudo ./install-systemd.sh"
   exit 1
fi

# Vérification que Docker est installé
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    exit 1
fi

# Vérification que docker compose est disponible
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose n'est pas disponible"
    exit 1
fi

echo "📝 Création des services systemd..."

# Service de démarrage
cat > /etc/systemd/system/immich-analyze-start.service << EOF
[Unit]
Description=Start Immich Analyze LlamaCPP
After=network.target docker.service
Requires=docker.service

[Service]
Type=oneshot
WorkingDirectory=$PROJECT_DIR
ExecStart=/usr/bin/docker compose up -d immich-analyze-llamacpp
User=$USER_NAME
Group=$USER_NAME
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF

# Service d'arrêt
cat > /etc/systemd/system/immich-analyze-stop.service << EOF
[Unit]
Description=Stop Immich Analyze LlamaCPP
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
WorkingDirectory=$PROJECT_DIR
ExecStart=/usr/bin/docker compose stop immich-analyze-llamacpp
User=$USER_NAME
Group=$USER_NAME
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF

# Timer de démarrage (minuit)
cat > /etc/systemd/system/immich-analyze-start.timer << EOF
[Unit]
Description=Start Immich Analyze at midnight
Requires=immich-analyze-start.service

[Timer]
OnCalendar=*-*-* 00:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Timer d'arrêt (6h)
cat > /etc/systemd/system/immich-analyze-stop.timer << EOF
[Unit]
Description=Stop Immich Analyze at 6am
Requires=immich-analyze-stop.service

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

echo "✅ Services systemd créés"

# Recharger systemd
echo "🔄 Rechargement de systemd..."
systemctl daemon-reload

# Activation des timers
echo "⚡ Activation des timers..."
systemctl enable immich-analyze-start.timer
systemctl enable immich-analyze-stop.timer

# Démarrage des timers
echo "🎬 Démarrage des timers..."
systemctl start immich-analyze-start.timer
systemctl start immich-analyze-stop.timer

echo ""
echo "🎉 Installation terminée avec succès !"
echo ""
echo "📊 Statut des timers:"
systemctl list-timers | grep immich-analyze || echo "⚠️  Aucun timer trouvé (normal si première installation)"

echo ""
echo "🔍 Commandes utiles:"
echo "  - Voir les timers: systemctl list-timers | grep immich-analyze"
echo "  - Voir les logs: journalctl -u immich-analyze-start.service"
echo "  - Statut: systemctl status immich-analyze-start.timer"
echo "  - Désinstaller: sudo ./uninstall-systemd.sh"

echo ""
echo "⏰ Votre service tournera automatiquement:"
echo "   • Démarrage: tous les jours à 00:00"
echo "   • Arrêt: tous les jours à 06:00"