const WORKFLOW_BY_CRON = {
  "17 * * * *": "shadow-forecast.yml",
  "43 4 * * *": "shadow-score.yml",
};

async function dispatchWorkflow(env, workflow) {
  const owner = env.GITHUB_OWNER;
  const repository = env.GITHUB_REPO;
  const ref = env.GITHUB_REF || "main";
  if (!owner || !repository || !env.GITHUB_TOKEN) {
    throw new Error("Missing GITHUB_OWNER, GITHUB_REPO, or GITHUB_TOKEN binding");
  }
  const response = await fetch(
    `https://api.github.com/repos/${owner}/${repository}/actions/workflows/${workflow}/dispatches`,
    {
      method: "POST",
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        "User-Agent": "sea-pm25-shadow-trigger/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
      },
      body: JSON.stringify({ ref }),
    },
  );
  if (response.status !== 204) {
    const detail = (await response.text()).slice(0, 500);
    throw new Error(`GitHub dispatch failed: HTTP ${response.status}: ${detail}`);
  }
}

export default {
  async scheduled(controller, env, context) {
    const workflow = WORKFLOW_BY_CRON[controller.cron];
    if (!workflow) {
      throw new Error(`No workflow mapped for cron ${controller.cron}`);
    }
    context.waitUntil(dispatchWorkflow(env, workflow));
  },

  async fetch() {
    return Response.json({
      service: "sea-pm25-deployment-shadow-trigger",
      status: "ok",
      schedules_utc: Object.keys(WORKFLOW_BY_CRON),
    });
  },
};
