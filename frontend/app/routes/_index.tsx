import type { MetaFunction } from "@remix-run/node";

export const meta: MetaFunction = () => {
  return [
    { title: "ResearchSynthesis" },
    { name: "description", content: "AI-powered research synthesis platform" },
  ];
};

export default function Index() {
  return (
    <div className="flex h-screen items-center justify-center">
      <div className="flex flex-col items-center gap-16">
        <header className="flex flex-col items-center gap-9">
          <h1 className="leading text-5xl font-bold text-gray-800 dark:text-gray-100">
            ResearchSynthesis
          </h1>
          <div className="h-[144px] w-[434px]">
            <div className="flex items-center justify-center rounded-full bg-blue-500 p-8 shadow-xl">
              <span className="text-4xl text-white font-semibold">Hello World</span>
            </div>
          </div>
        </header>
        <nav className="flex flex-col items-center justify-center gap-4 rounded-3xl border border-gray-200 p-6 dark:border-gray-700">
          <p className="leading-6 text-gray-700 dark:text-gray-200">
            Remix + Python FastAPI Stack
          </p>
          <ul className="flex gap-4">
            <li>
              <a
                className="group flex items-center gap-3 self-stretch p-3 leading-normal text-blue-700 hover:underline dark:text-blue-500"
                href="https://remix.run"
                target="_blank"
                rel="noreferrer"
              >
                Remix Docs
              </a>
            </li>
            <li>
              <a
                className="group flex items-center gap-3 self-stretch p-3 leading-normal text-blue-700 hover:underline dark:text-blue-500"
                href="https://fastapi.tiangolo.com"
                target="_blank"
                rel="noreferrer"
              >
                FastAPI Docs
              </a>
            </li>
          </ul>
        </nav>
      </div>
    </div>
  );
}
