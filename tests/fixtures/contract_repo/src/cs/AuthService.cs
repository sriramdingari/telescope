namespace Company.Product.Services
{
    public class AuthService
    {
        public bool Authenticate(string token)
        {
            return token.Length > 0;
        }
    }
}
