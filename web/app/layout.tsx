import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sky Tonight",
  description: "For any place in Spain and any night up to six months ahead: the chance of a clear sky, and what you can see.",
  // no favicon: an empty icon also stops the browser asking for /favicon.ico
  icons: { icon: "data:," },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
