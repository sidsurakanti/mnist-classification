import type { Metadata } from "next";
import { Poppins } from "next/font/google";
import "@/styles/globals.css";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"]
});


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
