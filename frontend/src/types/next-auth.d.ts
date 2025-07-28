import NextAuth from "next-auth";

declare module "next-auth" {
  /**
   * Extend the built-in user type
   */
  interface User {
    id: string;
    name?: string | null;
    email?: string | null;
    accessToken?: string;
    refreshToken?: string;
  }

  /**
   * Extend the built-in session type
   */
  interface Session {
    user: User;
    accessToken?: string;
    refreshToken?: string;
    error?: string;
  }
}

declare module "next-auth/jwt" {
  /**
   * Extend the built-in JWT types
   */
  interface JWT {
    id?: string;
    name?: string | null;
    email?: string | null;
    accessToken?: string;
    refreshToken?: string;
    error?: string;
  }
}
