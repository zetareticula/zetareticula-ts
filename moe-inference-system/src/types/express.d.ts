declare module 'express' {
  import { Application, Request, Response, NextFunction } from 'express-serve-static-core';
  const express: () => Application;
  export = express;
  export { Application, Request, Response, NextFunction };
}

declare module 'cors' {
  import { RequestHandler } from 'express';
  const cors: (options?: any) => RequestHandler;
  export = cors;
}
