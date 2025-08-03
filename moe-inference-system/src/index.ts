import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { initRoutes } from '@integration/api';
import { createLogger } from '@core/logger';
import { MoE } from '@core/moe/expert';
import { expert1, expert2 } from '@core/mock/experts';

// Load environment variables
dotenv.config({
  path: path.resolve(__dirname, '../../.env')
});

// Initialize application
const app = express();
const port = process.env.PORT || 3000;
const logger = createLogger('app');

// Initialize MoE with mock experts
const moe = new MoE([expert1, expert2]);
logger.info('Initialized Mixture of Experts system');

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Initialize routes
initRoutes(app, moe);

// Error handling middleware
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal Server Error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined,
  });
});

// Start server
const server = app.listen(port, () => {
  logger.info(`Server is running on port ${port}`);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason: Error | unknown) => {
  logger.error('Unhandled Rejection at:', reason);
  // Close server & exit process
  server.close(() => process.exit(1));
});

export default app;
