#!/bin/bash
# setup.sh - Déploiement complet d'immich-analyze avec services systemd

set -e

echo "🚀 Déploiement complet d'immich-analyze"
echo "======================================"
echo ""

# Étape 1: Construction et démarrage des containers
echo "📦 Étape 1: Construction et démarrage des containers Docker..."
docker compose down 2>/dev/null || true
docker compose up -d --build

# Vérifier que le container démarre correctement
echo "⏳ Vérification du démarrage du container..."
sleep 5

if docker compose ps | grep -q "immich-analyze-llamacpp.*Up"; then
    echo "✅ Container démarré avec succès"
else
    echo "⚠️  Container non démarré, mais on continue avec l'installation systemd"
fi

# Arrêter le container pour que systemd le gère
echo "⏹️  Arrêt du container (systemd va le gérer)"
docker compose stop

echo ""

# Étape 2: Installation des services systemd
echo "⚙️  Étape 2: Installation des services systemd..."

if [[ $EUID -eq 0 ]]; then
    # Déjà root
    ./install-systemd.sh
else
    # Demander sudo
    echo "🔐 Demande des permissions administrateur pour installer les services systemd..."
    sudo ./install-systemd.sh
fi

echo ""
echo "🎉 Déploiement terminé avec succès !"
echo "======================================"
echo ""

# Résumé
echo "📋 Résumé de l'installation:"
echo "   ✅ Container Docker construit et configuré"
echo "   ✅ Services systemd installés et activés"
echo "   ✅ Planification automatique configurée"
echo ""

echo "⏰ Planification:"
echo "   • Démarrage automatique: tous les jours à 00:00"
echo "   • Arrêt automatique: tous les jours à 06:00"
echo "   • Redémarrage en cas de crash entre 00:00-06:00"
echo ""

echo "🔍 Commandes utiles:"
echo "   • Voir les timers: systemctl list-timers | grep immich-analyze"
echo "   • Voir les logs: journalctl -u immich-analyze-start.service"
echo "   • Statut container: docker compose ps"
echo "   • Démarrer manuellement: docker compose up -d"
echo "   • Arrêter manuellement: docker compose stop"
echo "   • Désinstaller systemd: sudo ./uninstall-systemd.sh"
echo ""

# Afficher le statut actuel
echo "📊 Statut actuel des timers:"
systemctl list-timers | grep immich-analyze 2>/dev/null || echo "   (Timers installés, attente de la prochaine exécution)"

echo ""
echo "✨ Votre service d'analyse Immich est maintenant configuré pour fonctionner automatiquement !"