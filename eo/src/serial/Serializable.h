# ifndef __EOSERIAL_SERIALIZABLE_H__
# define __EOSERIAL_SERIALIZABLE_H__

# include <string>

namespace eoserial
{

class Object; // to avoid recursive inclusion with JsonObject

/**
 * @brief Interface showing that object can be written to a eoserial type
 * (currently JSON).
 */
class Printable
{
public:
    /**
     * @brief Serializes the object to JSON format.
     * @return A JSON object created with new.
     */
    virtual eoserial::Object* pack() const = 0;
};

/**
 * @brief Interface showing that object can be eoserialized (written and read
 * from an input).
 *
 * Note : Persistent objects should have a default non-arguments constructor.
 */
class Persistent : public Printable
{
    public:
    /**
     * @brief Loads class fields from a JSON object.
     * @param json A JSON object. Programmer doesn't have to delete it, it
     * is automatically done.
     */
    virtual void unpack(const eoserial::Object* json) = 0;
};

} // namespace eoserial

# endif // __EOSERIAL_SERIALIZABLE_H__
