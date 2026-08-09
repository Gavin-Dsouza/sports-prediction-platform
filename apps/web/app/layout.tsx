import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sports Prediction Platform",
  description: "Probabilistic MLB predictions and +EV betting intelligence.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>
        <div className="min-h-screen">
          <header className="border-b border-slate-800 px-6 py-4">
            <div className="mx-auto flex max-w-6xl items-center justify-between">
              <span className="text-lg font-semibold tracking-tight">
                Sports Prediction Platform <span className="text-slate-500">· MLB</span>
              </span>
              <nav className="flex gap-4 text-sm text-slate-400">
                <a href="/" className="hover:text-slate-100">
                  Today
                </a>
                <a href="/backtests" className="hover:text-slate-100">
                  Backtests
                </a>
                <a href="/3d" className="hover:text-slate-100">
                  3D View
                </a>
              </nav>
            </div>
          </header>
          <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
