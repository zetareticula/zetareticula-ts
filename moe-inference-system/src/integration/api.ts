import { Application } from 'actix-web';
import { ZetaVaultSynergy } from './zeta_vault';

export function initRoutes(app: Application, zeta: ZetaVaultSynergy): Application {
  app.post('/infer', async (req, res) => {
    const input = req.body as MoEInput;
    const result = await zeta.infer(input);
    res.json(result);
  });
  return app;
}