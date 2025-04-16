import type { Metadata } from "next";
import "@/styles/globals.css";
import { poppins } from "@/styles/fonts";


export const metadata: Metadata = {
  title: "Digit Recognition",
  description: "By Sid",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`h-screen ${poppins.className} subpixel-antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
