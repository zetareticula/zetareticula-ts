import winston from 'winston';

const { combine, timestamp, printf, colorize, json } = winston.format;

const logFormat = printf(({ level, message, timestamp, ...meta }) => {
  const metaString = Object.keys(meta).length ? ` ${JSON.stringify(meta)}` : '';
  return `${timestamp} [${level}]: ${message}${metaString}`;
});

export function createLogger(module: string) {
  const isDevelopment = process.env.NODE_ENV !== 'production';

  const logger = winston.createLogger({
    level: isDevelopment ? 'debug' : 'info', 
    format: combine(
      timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
      isDevelopment ? colorize() : winston.format.simple(),
      isDevelopment ? logFormat : json()
    ),
    defaultMeta: { module },
    transports: [
      new winston.transports.Console(),
      new winston.transports.File({ 
        filename: 'logs/error.log', 
        level: 'error' 
      }),
      new winston.transports.File({ 
        filename: 'logs/combined.log' 
      })
    ],
  });

  return {
    debug: (message: string, meta?: any) => logger.debug(message, meta),
    info: (message: string, meta?: any) => logger.info(message, meta),
    warn: (message: string, meta?: any) => logger.warn(message, meta),
    error: (message: string, meta?: any) => logger.error(message, meta),
  };
}

export type Logger = ReturnType<typeof createLogger>;
