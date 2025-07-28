import { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import { prisma } from "./prisma";
import { compare } from "bcryptjs";
import { sign } from "jsonwebtoken";

// Extended user type with accessToken
interface UserWithToken {
  id: string;
  email?: string | null;
  name?: string | null;
  accessToken?: string;
  refreshToken?: string;
}

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        // Find user in database
        const user = await prisma.user.findUnique({
          where: { email: credentials.email },
        });

        // If no user found or password doesn't match
        if (!user || !user.password) {
          return null;
        }

        // Compare password with stored hash
        const isPasswordValid = await compare(credentials.password, user.password);

        if (!isPasswordValid) {
          return null;
        }

        // Create a JWT token with the email as the sub claim
        // This matches what the backend expects
        const secret = process.env.JWT_SECRET_KEY || process.env.NEXTAUTH_SECRET;
        if (!secret) {
          throw new Error("JWT_SECRET_KEY or NEXTAUTH_SECRET is not defined");
        }

        const accessToken = sign(
          {
            sub: user.email, // Backend uses email as the user ID
            email: user.email,
            exp: Math.floor(Date.now() / 1000) + 24 * 60 * 60, // 24 hours expiration
            type: "access"
          },
          secret
        );

        const refreshToken = sign(
          {
            sub: user.email,
            email: user.email,
            exp: Math.floor(Date.now() / 1000) + 7 * 24 * 60 * 60, // 7 days expiration
            type: "refresh"
          },
          secret
        );

        // Return user without password
        return {
          id: user.id,
          email: user.email,
          name: user.name,
          accessToken: accessToken,
          refreshToken: refreshToken,
        } as UserWithToken;
      },
    }),
  ],
  pages: {
    signIn: "/login",
    error: "/login", // Redirect to login page on error
    signOut: "/", // Redirect to home page on signOut
  },
  session: {
    strategy: "jwt",
    maxAge: 24 * 60 * 60, // 24 hours
  },
  callbacks: {
    async jwt({ token, user }) {
      try {
        if (user) {
          token.id = user.id;
          token.email = user.email;
          token.name = user.name;
          // Store the access token in the JWT
          if ((user as UserWithToken).accessToken) {
            token.accessToken = (user as UserWithToken).accessToken;
          }
          // Store the refresh token in the JWT
          if ((user as UserWithToken).refreshToken) {
            token.refreshToken = (user as UserWithToken).refreshToken;
          }
        }
        return token;
      } catch (error) {
        console.error("Error in jwt callback:", error);
        // Return a minimal token to avoid errors
        return {
          name: "Invalid Session",
          email: "invalid@example.com",
          error: "invalid_token"
        };
      }
    },
    async session({ session, token }) {
      try {
        // Check if token has error flag
        if (token.error === "invalid_token") {
          // Return a session that indicates an error
          return {
            ...session,
            error: "invalid_token",
            expires: new Date(0).toISOString(), // Expired session
          };
        }

        if (token && session.user) {
          session.user.id = token.id as string;
          session.user.name = token.name as string;
          session.user.email = token.email as string;
          // Add the access token to the session
          (session as any).accessToken = token.accessToken;
          // Also add it to user for easier access
          (session.user as any).accessToken = token.accessToken;
          // Add the refresh token to the session
          (session as any).refreshToken = token.refreshToken;
          // Also add it to user for easier access
          (session.user as any).refreshToken = token.refreshToken;
        }
        return session;
      } catch (error) {
        console.error("Error in session callback:", error);
        // Return a minimal session to avoid errors
        return {
          ...session,
          error: "session_error",
          expires: new Date(0).toISOString(), // Expired session
        };
      }
    },
  },
  debug: process.env.NODE_ENV === "development",
  secret: process.env.NEXTAUTH_SECRET,
};