#!/bin/bash
set -e

echo "🚀 Starting deployment..."

# Install dependencies
echo "📦 Installing dependencies..."
npm ci --production

# Build the application
echo "🔨 Building the application..."
npm run build

# Stop existing instance if running
echo "🛑 Stopping existing instance..."
pm2 delete moe-inference-system 2> /dev/null || true

# Start the application with PM2
echo "🚀 Starting application with PM2..."
NODE_ENV=production pm2 start ecosystem.config.js --env production

# Save PM2 process list
echo "💾 Saving PM2 process list..."
pm2 save

# Set up PM2 to start on system boot
echo "🔌 Setting up PM2 startup..."
pm2 startup 2> /dev/null || true
pm2 save

echo "✅ Deployment complete!"
echo "📊 Check application status: pm2 status"
echo "📝 View logs: pm2 logs moe-inference-system"
echo "🌐 Application should be available at: http://localhost:3000"
