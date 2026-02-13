#!/bin/bash
# uninstall-systemd.sh - Désinstallation des services systemd pour immich-analyze

set -e

echo "🗑️  Désinstallation des services systemd pour immich-analyze..."

# Vérification des permissions
if [[ $EUID -ne 0 ]]; then
   echo "❌ Ce script doit être exécuté avec sudo"
   echo "💡 Usage: sudo ./uninstall-systemd.sh"
   exit 1
fi

echo "⏹️  Arrêt des timers..."
systemctl stop immich-analyze-start.timer 2>/dev/null || echo "   Timer start déjà arrêté"
systemctl stop immich-analyze-stop.timer 2>/dev/null || echo "   Timer stop déjà arrêté"

echo "🚫 Désactivation des services..."
systemctl disable immich-analyze-start.timer 2>/dev/null || echo "   Timer start déjà désactivé"
systemctl disable immich-analyze-stop.timer 2>/dev/null || echo "   Timer stop déjà désactivé"

echo "🗂️  Suppression des fichiers de service..."
rm -f /etc/systemd/system/immich-analyze-start.service
rm -f /etc/systemd/system/immich-analyze-stop.service
rm -f /etc/systemd/system/immich-analyze-start.timer
rm -f /etc/systemd/system/immich-analyze-stop.timer

echo "🔄 Rechargement de systemd..."
systemctl daemon-reload

echo ""
echo "✅ Désinstallation terminée !"
echo "📊 Services restants:"
systemctl list-timers | grep immich-analyze || echo "   Aucun service immich-analyze trouvé (désinstallation réussie)"

echo ""
echo "💡 Pour réinstaller les services:"
echo "   sudo ./install-systemd.sh"