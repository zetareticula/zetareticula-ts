import { Request, Response, NextFunction, Application } from 'express';
import { body, validationResult } from 'express-validator';
import { StatusCodes } from 'http-status-codes';
import { createLogger } from '../core/logger';
import { MoE } from '../core/moe/expert';
import { MoEInput } from '../core/moe/types';

const logger = createLogger('api');

// Input validation middleware
const validateInferRequest = [
  body('features')
    .isArray({ min: 1 })
    .withMessage('Features must be a non-empty array')
    .custom((value: any[]) => value.every(Number.isFinite))
    .withMessage('All features must be numbers'),
  body('hardwareClass')
    .isIn(['cpu', 'gpu', 'tpu'])
    .withMessage('Invalid hardware class')
];

export function initRoutes(app: Application, moe: MoE): Application {
  /**
   * @route POST /infer
   * @desc Route for inference using Mixture of Experts
   * @access Public
   */
  app.post(
    '/infer',
    validateInferRequest,
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        // Check for validation errors
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
          return res.status(StatusCodes.BAD_REQUEST).json({ 
            error: 'Validation Error',
            details: errors.array() 
          });
        }

        const input: MoEInput = {
          features: new Float32Array(req.body.features),
          hardwareClass: req.body.hardwareClass
        };

        logger.info('Processing inference request', { 
          inputLength: input.features.length,
          hardwareClass: input.hardwareClass 
        });

        const result = await moe.routeInput(input);
        
        return res.status(StatusCodes.OK).json({
          success: true,
          data: result
        });
      } catch (error) {
        logger.error('Inference error:', error);
        return next(error);
      }
    }
  );

  // Health check endpoint
  app.get('/health', (_req: Request, res: Response) => {
    res.status(StatusCodes.OK).json({
      status: 'ok',
      timestamp: new Date().toISOString(),
      uptime: process.uptime()
    });
  });

  // 404 handler
  app.use((req: Request, res: Response) => {
    res.status(StatusCodes.NOT_FOUND).json({
      error: 'Not Found',
      message: `Cannot ${req.method} ${req.path}`
    });
  });

  return app;
}