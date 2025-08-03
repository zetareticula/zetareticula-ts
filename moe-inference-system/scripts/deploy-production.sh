#!/bin/bash
set -e

echo "🚀 Starting production deployment..."

# Clean up previous build
echo "🧹 Cleaning up previous build..."
rm -rf dist/
rm -f package-lock.json

# Install all dependencies (including dev dependencies for building)
echo "📦 Installing dependencies..."
npm install

# Build the application using default tsconfig to ensure node types are present
echo "🔨 Building the application (default tsconfig)..."
npx tsc -p tsconfig.json

# Remove dev-dependencies for the runtime image
echo "🧹 Pruning dev dependencies..."
npm prune --production

# Install only production dependencies for runtime
echo "📦 Installing production dependencies..."
npm ci --only=production

# Ensure the logs directory exists
echo "📁 Setting up logs directory..."
mkdir -p logs

# Stop existing instance if running
echo "🛑 Stopping existing instance..."
pm2 delete moe-inference-system 2> /dev/null || true

# Start the application with PM2 using the server.js file
echo "🚀 Starting application with PM2..."
NODE_ENV=production pm2 start server.js --name moe-inference-system --output logs/app.log --error logs/error.log --time

# Save PM2 process list
echo "💾 Saving PM2 process list..."
pm2 save

# Set up PM2 to start on system boot
echo "🔌 Setting up PM2 startup..."
pm2 startup 2> /dev/null || true
pm2 save

echo ""
echo "✅ Deployment complete!"
echo "📊 Check application status: pm2 status"
echo "📝 View logs: pm2 logs moe-inference-system"
echo "🌐 Application should be available at: http://localhost:3000"
echo ""
echo "To monitor the application:"
echo "  pm2 monit"
echo ""
echo "To view logs in real-time:"
echo "  pm2 logs moe-inference-system --lines 100"
echo ""
echo "To stop the application:"
echo "  pm2 stop moe-inference-system"
echo ""
echo "To restart the application:"
echo "  pm2 restart moe-inference-system"
