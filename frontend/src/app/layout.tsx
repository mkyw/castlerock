import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "@/app/providers";
import SessionProviderWrapper from "@/components/providers/SessionProviderWrapper";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Castlerock AI - AI-Powered Customer Support",
  description: "Transform your customer service with our AI-powered chatbot solution. Provide instant, accurate responses to customer inquiries 24/7.",
  keywords: ["AI chatbot", "customer support", "AI assistant", "business automation", "customer service"],
};

export const viewport = {
  themeColor: '#4f46e5',
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full bg-gray-50" suppressHydrationWarning>
      <body className={`${inter.className} antialiased h-full`}>
        <SessionProviderWrapper>
          <Providers>
            {children}
          </Providers>
        </SessionProviderWrapper>
      </body>
    </html>
  );
}
