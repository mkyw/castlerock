import { NextApiRequest, NextApiResponse, NextApiHandler } from 'next';
import { getSession } from 'next-auth/react';
import { Session } from 'next-auth';

/**
 * Higher-order function to protect API routes with authentication
 * @param handler The API route handler to protect
 * @returns A protected API route handler
 */
export const withAuth = (handler: NextApiHandler) => async (
  req: NextApiRequest,
  res: NextApiResponse
) => {
  // Get the session
  const session = await getSession({ req }) as Session & { accessToken?: string };

  // Check if user is authenticated
  if (!session?.user) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  // Check if access token exists
  if (!session.accessToken) {
    return res.status(401).json({ error: 'No access token found' });
  }

  // Add the access token to the request headers
  if (req.headers) {
    req.headers['Authorization'] = `Bearer ${session.accessToken}`;
  }

  // Call the API route handler
  return handler(req, res);
};
