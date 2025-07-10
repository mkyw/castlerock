import { PrismaAdapter } from "@next-auth/prisma-adapter";
import { NextAuthOptions, Session, DefaultSession } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import GoogleProvider from "next-auth/providers/google";
import { compare } from "bcryptjs";
import { prisma } from "@/lib/prisma";

declare module "next-auth" {
  interface Session extends DefaultSession {
    user: {
      id: string;
      name?: string | null;
      email?: string | null;
      image?: string | null;
      accessToken?: string;
    } & DefaultSession['user'];
  }

  interface User {
    id: string;
    name?: string | null;
    email?: string | null;
    image?: string | null;
    accessToken?: string;
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    id: string;
    name?: string | null;
    email?: string | null;
    picture?: string | null;
    accessToken?: string;
  }
}

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(prisma),
  session: {
    strategy: "jwt",
  },
  pages: {
    signIn: "/login",
    signOut: "/login",
    error: "/login",
  },
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          throw new Error("Invalid credentials");
        }

        const user = await prisma.user.findUnique({
          where: { email: credentials.email },
        });

        if (!user || !user.password) {
          throw new Error("No user found with this email");
        }

        const isPasswordValid = await compare(credentials.password, user.password);

        if (!isPasswordValid) {
          throw new Error("Invalid password");
        }

        return {
          id: user.id,
          email: user.email,
          name: user.name,
          image: user.image || null,
        };
      },
    }),
  ],

  callbacks: {
    async session({ session, token }): Promise<Session> {
      console.log('Session callback - token:', token);
      if (token) {
        // Add user info to session
        session.user = {
          ...session.user,
          id: token.id as string,
          name: token.name || null,
          email: token.email || null,
          image: token.picture || null,
        };
        // Add access token to session
        session.accessToken = token.accessToken as string;
        console.log('Session callback - session with token:', session);
      }
      return session;
    },
    async jwt({ token, user, account }) {
      console.log('JWT callback - token, user, account:', { token, user, account });
      
      // Initial sign in
      if (account && user) {
        console.log('Initial sign in - account and user:', { account, user });
        // Add access token to JWT
        token.accessToken = account.access_token || account.id_token || `generated-${Date.now()}`;
        token.id = user.id;
        token.name = user.name;
        token.email = user.email;
        token.picture = user.image;
        console.log('JWT callback - updated token:', token);
      }
      return token;
    },
  },
  secret: process.env.NEXTAUTH_SECRET,
  debug: process.env.NODE_ENV === "development",
} as const;
