import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  const testUser = {
    email: `test_${Date.now()}@example.com`,
    password: 'Test123!',
    name: 'Test User',
    firstName: 'Test',
    lastName: 'User'
  };

  test('should allow user to sign up and log in', async ({ page }) => {
    // Enable request logging
    page.on('request', request => console.log('>>', request.method(), request.url()));
    page.on('response', response => 
      console.log('<<', response.status(), response.url())
    );

    console.log('1. Navigating to signup page...');
    await page.goto('/signup');
    
    console.log('2. Filling out signup form...');
    await page.waitForSelector('input[name="firstName"]', { state: 'visible' });
    
    // Fill out the signup form with more reliable selectors
    await page.fill('input[name="firstName"]', testUser.firstName);
    await page.fill('input[name="lastName"]', testUser.lastName);
    await page.fill('input[type="email"]', testUser.email);
    await page.fill('input[type="password"]:nth-of-type(1)', testUser.password);
    await page.fill('input[type="password"]:nth-of-type(2)', testUser.password);
    await page.check('input[type="checkbox"]');
    
    console.log('3. Submitting signup form...');
    // Submit the form and wait for navigation
    const [response] = await Promise.all([
      page.waitForResponse(response => {
        const isAuthResponse = response.url().includes('/api/auth/') || 
                             response.url().includes('/api/auth/callback/');
        console.log('Auth response:', response.status(), response.url());
        return isAuthResponse;
      }),
      page.click('button[type="submit"]')
    ]);

    console.log('4. Checking for successful signup...');
    // Wait for navigation to complete
    await page.waitForURL('**/login', { timeout: 10000 });
    
    console.log('5. Filling out login form...');
    await page.waitForSelector('input[type="email"]', { state: 'visible' });
    await page.fill('input[type="email"]', testUser.email);
    await page.fill('input[type="password"]', testUser.password);
    
    console.log('6. Submitting login form...');
    await Promise.all([
      page.waitForResponse(response => {
        const isAuthResponse = response.url().includes('/api/auth/callback/credentials');
        if (isAuthResponse) {
          console.log('Login response:', response.status(), response.url());
        }
        return isAuthResponse;
      }),
      page.click('button[type="submit"]')
    ]);

    console.log('7. Verifying login...');
    // Wait for navigation after login
    await page.waitForURL('**/dashboard', { timeout: 10000 });
    
    // Verify we're logged in by checking for user's name in the navbar
    const navBar = page.locator('nav');
    await expect(navBar).toContainText(testUser.firstName, { timeout: 5000 });
    
    console.log('8. Verifying session cookie...');
    // Verify we have a session cookie
    const cookies = await page.context().cookies();
    const sessionCookie = cookies.find(cookie => 
      cookie.name.startsWith('next-auth.session-token') || 
      cookie.name.startsWith('__Secure-next-auth.session-token')
    );
    
    console.log('Found cookies:', cookies.map(c => c.name));
    expect(sessionCookie).toBeTruthy();
    
    console.log('✅ Authentication test completed successfully!');
  });
});
